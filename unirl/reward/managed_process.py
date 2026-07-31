"""Rank-affine reward backend backed by one managed local child process.

Each train Worker constructs one backend, so the child naturally inherits that
Worker's ``CUDA_VISIBLE_DEVICES`` and scores only the local DP shard. The child
uses an explicit Python executable, isolating EditReward dependencies from the
train/vLLM environment without asking Ray for another logical GPU slot.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from unirl.config.require import require
from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec

logger = logging.getLogger(__name__)


@dataclass
class ManagedRewardProcessSpec(BaseRewardComponentSpec):
    # Accepted for drop-in Hydra replacement of RemoteRewardSpec; ignored
    # because each rank allocates its own loopback endpoint dynamically.
    base_url: str = ""
    python_executable: str = ""
    service_root: str = ""
    checkpoint_path: str = ""
    model_name_or_path: str = ""
    config_path: Optional[str] = None
    scorer_name: str = "editreward"
    device: str = "cuda"
    dtype: str = "bfloat16"
    rm_head_type: str = "ranknet_multi_head"
    startup_timeout: float = 1200.0
    shutdown_timeout: float = 30.0
    log_dir: str = "/tmp"

    required_rewards: Tuple[str, ...] = ("editreward",)
    reward_weights: Optional[Dict[str, float]] = None
    batch_size: int = 8
    timeout: float = 600.0
    max_retries: int = 1
    retry_delay: float = 0.0
    sub_metric_reduce: str = "mean"
    aggregation_method: str = "weighted_sum"
    image_format: str = "JPEG"
    image_quality: int = 95
    raise_on_failure: bool = True

    def __post_init__(self) -> None:
        require(Path(self.python_executable).is_file(), f"reward python not found: {self.python_executable}")
        require(Path(self.service_root).is_dir(), f"reward service root not found: {self.service_root}")
        require(Path(self.checkpoint_path).is_dir(), f"reward checkpoint not found: {self.checkpoint_path}")
        require(Path(self.model_name_or_path).is_dir(), f"reward base model not found: {self.model_name_or_path}")
        require(self.startup_timeout > 0, "startup_timeout must be positive")
        require(self.shutdown_timeout > 0, "shutdown_timeout must be positive")
        require(tuple(self.required_rewards) == (self.scorer_name,), "managed process serves exactly scorer_name")


class ManagedRewardProcessBackend(RemoteRewardBackend):
    """Start one persistent reward server in this Worker's reward environment."""

    def __init__(self, *, config: ManagedRewardProcessSpec, base_device: str) -> None:
        self.process_config = config
        self._process: Optional[subprocess.Popen] = None
        self._process_log = None
        self._disposed = False
        try:
            base_url = self._start_child()
        except Exception:
            self._stop_child()
            raise
        remote_spec = RemoteRewardSpec(
            base_url=base_url,
            required_rewards=tuple(config.required_rewards),
            reward_weights=dict(config.reward_weights or {}),
            batch_size=int(config.batch_size),
            timeout=float(config.timeout),
            max_retries=int(config.max_retries),
            retry_delay=float(config.retry_delay),
            sub_metric_reduce=str(config.sub_metric_reduce),
            aggregation_method=str(config.aggregation_method),
            image_format=str(config.image_format),
            image_quality=int(config.image_quality),
            input_kind="image",
            raise_on_failure=bool(config.raise_on_failure),
        )
        super().__init__(config=remote_spec, base_device=base_device)
        atexit.register(self._stop_child)

    def _start_child(self) -> str:
        cfg = self.process_config
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.set_inheritable(True)
        port = int(listener.getsockname()[1])

        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"editreward-child-{os.getpid()}-{port}.log"
        self._process_log = log_path.open("ab", buffering=0)

        params = {
            "checkpoint_path": cfg.checkpoint_path,
            "config_path": cfg.config_path,
            "model_name_or_path": cfg.model_name_or_path,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "rm_head_type": cfg.rm_head_type,
            "offload_between_calls": False,
        }
        command = [
            cfg.python_executable,
            "-m",
            "reward_service.direct_server",
            "--fd",
            str(listener.fileno()),
            "--scorer",
            cfg.scorer_name,
            "--params-json",
            json.dumps(params, separators=(",", ":")),
        ]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = cfg.service_root + (f":{existing_pythonpath}" if existing_pythonpath else "")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env.pop("RAY_ADDRESS", None)
        env["UNIRL_REWARD_PARENT_PID"] = str(os.getpid())

        logger.info(
            "starting rank-affine reward child port=%d cuda_visible=%s python=%s log=%s",
            port,
            env.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            cfg.python_executable,
            log_path,
        )
        try:
            self._process = subprocess.Popen(
                command,
                env=env,
                pass_fds=(listener.fileno(),),
                stdout=self._process_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            listener.close()

        base_url = f"http://127.0.0.1:{port}"
        session = requests.Session()
        session.trust_env = False
        deadline = time.monotonic() + float(cfg.startup_timeout)
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"reward child exited during startup with code {self._process.returncode}; log={log_path}"
                    )
                try:
                    response = session.get(f"{base_url}/health", timeout=2.0)
                    if response.status_code == 200:
                        body = response.json()
                        if cfg.scorer_name in dict(body.get("rewards") or {}):
                            logger.info("rank-affine reward child ready at %s", base_url)
                            return base_url
                except requests.RequestException:
                    pass
                time.sleep(1.0)
        finally:
            session.close()
        self._stop_child()
        raise TimeoutError(f"reward child did not become ready within {cfg.startup_timeout}s; log={log_path}")

    def is_available(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        return super().is_available()

    def offload(self) -> None:
        """Resident mode: keep the reward model on its assigned GPU."""

    def onload(self) -> None:
        """Resident mode: the reward model is already on its assigned GPU."""

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        super().dispose()
        self._stop_child()

    def _stop_child(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=float(self.process_config.shutdown_timeout))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self._process_log is not None:
            self._process_log.close()
            self._process_log = None

    def __del__(self) -> None:
        try:
            self._stop_child()
        except Exception:
            pass


__all__ = ["ManagedRewardProcessBackend", "ManagedRewardProcessSpec"]

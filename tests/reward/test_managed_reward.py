from __future__ import annotations

import base64
import io
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import reward_service.direct_server as direct_server
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image
from reward_service.schemas import RewardRequest as WireRewardRequest
from reward_service.scorers.base import BaseScorer
from reward_service.scorers.editreward import EditRewardScorer

from unirl.reward.managed_process import (
    ManagedProcessConfig,
    ManagedScorerConfig,
    ManagedScorerProcessBackend,
    ManagedScorerProcessSpec,
    _validate_visible_device,
)
from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec
from unirl.reward.service import RewardService
from unirl.types.primitives import Images, Texts
from unirl.types.reward import RewardRequest


def _wire_request(index: int, *, scorer_version: str | None = None) -> dict:
    return {
        "history": [{"text": f"p{index}", "image_b64": "unused"}],
        "required_rewards": ["fake"],
        "metadata": None,
        "request_id": f"request-{index}",
        "sample_id": f"sample-{index}",
        "group_id": f"group-{index // 2}",
        "source_rank": 3,
        "policy_version": 7,
        "scorer_version": scorer_version,
        "idempotency_key": f"key-{index}",
    }


def _response_for(chunk: list[dict], *, scorer_version: str = "1") -> dict:
    return {
        "protocol_version": "1",
        "results": [{"fake": {"score": float(item["request_id"].split("-")[-1])}} for item in chunk],
        "errors": [{} for _ in chunk],
        "identities": [
            {
                **{
                    key: item.get(key)
                    for key in (
                        "request_id",
                        "sample_id",
                        "group_id",
                        "source_rank",
                        "policy_version",
                        "idempotency_key",
                    )
                },
                "scorer_version": scorer_version,
            }
            for item in chunk
        ],
    }


def test_remote_request_chunking_preserves_order() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(
            base_url="http://unused",
            required_rewards=("fake",),
            request_batch_size=2,
            require_identity_echo=True,
            expected_scorer_version="1",
        ),
        base_device="cpu",
    )
    calls: list[list[str]] = []

    def post(payload: dict) -> dict:
        chunk = payload["requests"]
        calls.append([item["request_id"] for item in chunk])
        return _response_for(chunk)

    backend._post_score = post
    wire = [_wire_request(index, scorer_version="1") for index in range(5)]
    merged = backend._post_score_requests(wire)

    assert calls == [["request-0", "request-1"], ["request-2", "request-3"], ["request-4"]]
    assert [row["fake"]["score"] for row in merged["results"]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [row["request_id"] for row in merged["identities"]] == [f"request-{index}" for index in range(5)]
    backend.dispose()


def test_remote_legacy_default_keeps_one_post() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(base_url="http://unused", required_rewards=("fake",)),
        base_device="cpu",
    )
    calls: list[int] = []
    backend._post_score = lambda payload: calls.append(len(payload["requests"])) or _response_for(payload["requests"])

    backend._post_score_requests([_wire_request(index) for index in range(5)])
    assert calls == [5]
    backend.dispose()


def test_remote_rejects_identity_mismatch() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(
            base_url="http://unused",
            required_rewards=("fake",),
            require_identity_echo=True,
        ),
        base_device="cpu",
    )
    raw = _response_for([_wire_request(0)])
    raw["identities"][0]["sample_id"] = "wrong"
    backend._post_score = lambda payload: raw

    with pytest.raises(ValueError, match="identity mismatch"):
        backend._post_score_requests([_wire_request(0)])
    backend.dispose()


def test_remote_rejects_explicit_protocol_mismatch() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(base_url="http://unused", required_rewards=("fake",)),
        base_device="cpu",
    )
    backend._post_score = lambda payload: {
        **_response_for(payload["requests"]),
        "protocol_version": "2",
    }
    with pytest.raises(ValueError, match="protocol_version"):
        backend._post_score_requests([_wire_request(0)])
    backend.dispose()


def test_full_server_cannot_claim_requested_scorer_version() -> None:
    request = WireRewardRequest(
        history=[{"text": "prompt", "image_b64": _image_b64()}],
        required_rewards=["fake"],
        scorer_version="requested-version",
    )
    assert request.identity().scorer_version is None
    assert request.identity(actual_scorer_version="actual-version").scorer_version == "actual-version"


def test_remote_rejects_empty_submetrics() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(base_url="http://unused", required_rewards=("fake",)),
        base_device="cpu",
    )
    with pytest.raises(ValueError, match="empty sub-metric"):
        backend._reduce_sub_metrics({})
    backend.dispose()


def test_remote_compute_rewards_chunks_typed_image_request() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(
            base_url="http://unused",
            required_rewards=("fake",),
            request_batch_size=2,
            require_identity_echo=True,
        ),
        base_device="cpu",
    )
    calls: list[int] = []

    def post(payload: dict) -> dict:
        calls.append(len(payload["requests"]))
        return {
            "results": [{"fake": {"score": float(index + 1)}} for index, _ in enumerate(payload["requests"])],
            "errors": [{} for _ in payload["requests"]],
            "identities": [
                {
                    key: request.get(key)
                    for key in (
                        "request_id",
                        "sample_id",
                        "group_id",
                        "source_rank",
                        "policy_version",
                        "scorer_version",
                        "idempotency_key",
                    )
                }
                for request in payload["requests"]
            ],
        }

    backend._post_score = post
    request = RewardRequest(
        primitives={"text": Texts(texts=["a", "b", "c"])},
        generated={"image": Images(pixels=torch.rand(3, 3, 8, 8))},
        sample_ids=["s0", "s1", "s2"],
        group_ids=["g0", "g1", "g2"],
    )
    response = backend.compute_rewards(request)

    assert calls == [2, 1]
    assert response.successes == [True, True, True]
    assert response.rewards == [1.0, 2.0, 1.0]
    backend.dispose()


class _FakeScorer(BaseScorer):
    name = "fake"
    version = "test-v1"
    input_kind = "image"
    supports_offload = True
    sub_metric_names = ("score",)
    active = 0
    max_active = 0
    calls = 0
    guard = threading.Lock()

    def __init__(self, *, delay: float = 0.0, return_nan: bool = False) -> None:
        self.delay = delay
        self.return_nan = return_nan
        self.moves: list[str] = []

    def score(self, items):
        with self.guard:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            type(self).calls += 1
        time.sleep(self.delay)
        with self.guard:
            type(self).active -= 1
        value = math.nan if self.return_nan else 1.0
        return [{"score": value} for _ in items]

    def onload(self) -> None:
        self.moves.append("onload")

    def offload(self) -> None:
        self.moves.append("offload")

    def drain(self) -> None:
        self.moves.append("drain")


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _score_payload(*, key: str = "key", scorer_version: str = "test-v1") -> dict:
    return {
        "protocol_version": "1",
        "requests": [
            {
                "history": [{"text": "prompt", "image_b64": _image_b64()}],
                "required_rewards": ["fake"],
                "request_id": "request",
                "sample_id": "sample",
                "group_id": "group",
                "source_rank": 0,
                "policy_version": 1,
                "scorer_version": scorer_version,
                "idempotency_key": key,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset_fake_scorer() -> None:
    _FakeScorer.active = 0
    _FakeScorer.max_active = 0
    _FakeScorer.calls = 0


def test_direct_server_echoes_identity_and_caches_idempotent_result(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        first = client.post("/score", json=_score_payload())
        second = client.post("/score", json=_score_payload())

    assert first.status_code == second.status_code == 200
    assert first.json()["identities"][0]["sample_id"] == "sample"
    assert first.json()["identities"][0]["scorer_version"] == "test-v1"
    assert first.json()["results"] == [{"fake": {"score": 1.0}}]
    assert second.json()["results"] == first.json()["results"]
    assert _FakeScorer.calls == 1


def test_direct_server_rejects_idempotency_key_reuse_for_changed_payload(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    changed = _score_payload()
    changed["requests"][0]["history"][0]["text"] = "different prompt"
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        assert client.post("/score", json=_score_payload()).status_code == 200
        response = client.post("/score", json=changed)

    assert response.status_code == 409
    assert "different payload" in response.json()["detail"]


def test_direct_server_rejects_conflicting_keys_inside_one_batch(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    first = _score_payload()["requests"][0]
    second = {**first, "history": [{"text": "different prompt", "image_b64": _image_b64()}]}
    payload = {"protocol_version": "1", "requests": [first, second]}
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        response = client.post("/score", json=payload)

    assert response.status_code == 409
    assert "one batch" in response.json()["detail"]


def test_direct_server_deduplicates_identical_keys_inside_one_batch(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    request = _score_payload()["requests"][0]
    payload = {"protocol_version": "1", "requests": [request, dict(request)]}
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        response = client.post("/score", json=payload)

    assert response.status_code == 200
    assert response.json()["results"] == [{"fake": {"score": 1.0}}, {"fake": {"score": 1.0}}]
    assert _FakeScorer.calls == 1


def test_direct_server_converts_non_finite_result_to_error(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    with TestClient(direct_server.create_direct_app("fake", {"return_nan": True})) as client:
        response = client.post("/score", json=_score_payload())

    body = response.json()
    assert body["results"] == [{}]
    assert "non-finite" in body["errors"][0]["fake"]


def test_direct_server_rejects_empty_metric_map(monkeypatch) -> None:
    class EmptyScorer(_FakeScorer):
        def score(self, items):
            return [{} for _ in items]

    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: EmptyScorer)
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        response = client.post("/score", json=_score_payload())

    body = response.json()
    assert body["results"] == [{}]
    assert "omitted required metrics" in body["errors"][0]["fake"]


def test_direct_server_serializes_concurrent_calls(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    with TestClient(direct_server.create_direct_app("fake", {"delay": 0.05})) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda index: client.post("/score", json=_score_payload(key=f"key-{index}")),
                    range(2),
                )
            )
    assert all(response.status_code == 200 for response in responses)
    assert _FakeScorer.max_active == 1


def test_direct_server_lifecycle_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(direct_server, "get_scorer_cls", lambda name: _FakeScorer)
    with TestClient(direct_server.create_direct_app("fake", {})) as client:
        assert client.post("/lifecycle/offload").json()["state"] == "offloaded"
        assert client.post("/score", json=_score_payload()).status_code == 409
        assert client.post("/lifecycle/onload").json()["state"] == "resident"
        assert client.post("/score", json=_score_payload()).status_code == 200


def test_visible_gpu_validation() -> None:
    _validate_visible_device({"CUDA_VISIBLE_DEVICES": "3"}, allow_multiple=False, device="cuda")
    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        _validate_visible_device({"CUDA_VISIBLE_DEVICES": "2,3"}, allow_multiple=False, device="cuda")
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES"):
        _validate_visible_device({}, allow_multiple=False, device="cuda")
    _validate_visible_device({}, allow_multiple=False, device="cpu")


def test_reward_service_exposes_worker_teardown_hook() -> None:
    disposed: list[bool] = []
    backend = SimpleNamespace(
        get_model_name=lambda: "fake",
        dispose=lambda: disposed.append(True),
    )
    service = RewardService(backend=backend)
    service.shutdown()
    assert disposed == [True]


def test_per_call_lifecycle_serializes_onload_score_offload(monkeypatch) -> None:
    events: list[str] = []
    backend = ManagedScorerProcessBackend.__new__(ManagedScorerProcessBackend)
    backend.process_config = SimpleNamespace(lifecycle="per_call")
    backend._lifecycle_lock = threading.RLock()
    backend.onload = lambda: events.append("onload")
    backend.offload = lambda: events.append("offload")

    def score(_self, request):
        events.append("score.start")
        time.sleep(0.02)
        events.append("score.end")
        return SimpleNamespace()

    monkeypatch.setattr(RemoteRewardBackend, "compute_rewards", score)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: backend.compute_rewards(SimpleNamespace()), range(2)))

    assert events == [
        "onload",
        "score.start",
        "score.end",
        "offload",
        "onload",
        "score.start",
        "score.end",
        "offload",
    ]


def test_dispose_waits_for_inflight_per_call_score(monkeypatch) -> None:
    events: list[str] = []
    backend = ManagedScorerProcessBackend.__new__(ManagedScorerProcessBackend)
    backend.process_config = SimpleNamespace(
        lifecycle="per_call",
        process=SimpleNamespace(shutdown_timeout=1.0),
    )
    backend._lifecycle_lock = threading.RLock()
    backend._disposed = False
    backend._atexit_registered = False
    backend._session = SimpleNamespace(close=lambda: events.append("session.close"))
    backend.onload = lambda: events.append("onload")
    backend.offload = lambda: events.append("offload")
    backend._lifecycle = lambda action, **kwargs: events.append(action) or True
    backend._stop_child = lambda: events.append("child.stop")

    def score(_self, request):
        events.append("score.start")
        time.sleep(0.03)
        events.append("score.end")
        return SimpleNamespace()

    monkeypatch.setattr(RemoteRewardBackend, "compute_rewards", score)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scoring = pool.submit(backend.compute_rewards, SimpleNamespace())
        time.sleep(0.005)
        disposing = pool.submit(backend.dispose)
        scoring.result()
        disposing.result()

    assert events == [
        "onload",
        "score.start",
        "score.end",
        "offload",
        "shutdown",
        "session.close",
        "child.stop",
    ]


def test_editreward_exposes_explicit_device_lifecycle(monkeypatch) -> None:
    moves: list[str] = []

    class Model:
        def to(self, device: str):
            moves.append(device)
            return self

    scorer = EditRewardScorer.__new__(EditRewardScorer)
    scorer._target_device = "cuda"
    scorer.inferencer = type("Inferencer", (), {"model": Model(), "device": "cpu"})()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    scorer.onload()
    assert scorer.inferencer.device == "cuda"
    scorer.offload()
    assert scorer.inferencer.device == "cpu"
    assert moves == ["cuda", "cpu"]


def test_managed_spec_is_generic_and_image_scoped(tmp_path: Path) -> None:
    process = ManagedProcessConfig(
        python_executable=sys.executable,
        service_root=str(tmp_path),
    )
    scorer = ManagedScorerConfig(name="pickscore", input_kind="image", params={"device": "cpu"})
    client = RemoteRewardSpec(
        base_url="managed://rank-affine",
        required_rewards=("pickscore",),
        input_kind="image",
    )
    spec = ManagedScorerProcessSpec(process=process, scorer=scorer, client=client)
    assert spec.scorer.name == "pickscore"

    with pytest.raises(ValueError, match="image/image_edit"):
        ManagedScorerProcessSpec(
            process=process,
            scorer=ManagedScorerConfig(name="videoalign", input_kind="video", params={"device": "cpu"}),
            client=RemoteRewardSpec(
                base_url="managed://rank-affine",
                required_rewards=("videoalign",),
                input_kind="video",
            ),
        )


def test_managed_spec_resolves_hydra_scorer_params(tmp_path: Path) -> None:
    spec = ManagedScorerProcessSpec(
        process=ManagedProcessConfig(
            python_executable=sys.executable,
            service_root=str(tmp_path),
        ),
        scorer=ManagedScorerConfig(
            name="pickscore",
            input_kind="image",
            params=OmegaConf.create({"device": "cpu", "nested": {"value": 3}}),
        ),
        client=RemoteRewardSpec(
            base_url="managed://rank-affine",
            required_rewards=("pickscore",),
            input_kind="image",
        ),
    )
    assert spec.scorer.params == {"device": "cpu", "nested": {"value": 3}}


def test_managed_backend_launches_and_cleans_rank_local_child(tmp_path: Path) -> None:
    package = tmp_path / "reward_service"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "direct_server.py").write_text(
        """
import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--fd", type=int, required=True)
parser.add_argument("--scorer", required=True)
parser.add_argument("--input-kind", required=True)
parser.add_argument("--cache-size")
parser.add_argument("--params-json")
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def _write(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._write({
            "status": "ok",
            "state": "resident",
            "rewards": {args.scorer: ["resident"]},
            "scorer": {
                "name": args.scorer,
                "version": "1",
                "input_kind": "image",
                "supports_offload": True,
            },
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/score":
            requests = body.get("requests", [])
            identities = []
            for request in requests:
                identity = {
                    key: request.get(key)
                    for key in (
                        "request_id", "sample_id", "group_id", "source_rank",
                        "policy_version", "idempotency_key",
                    )
                }
                identity["scorer_version"] = "1"
                identities.append(identity)
            self._write({
                "protocol_version": "1",
                "results": [{args.scorer: {"score": 1.0}} for _ in requests],
                "errors": [{} for _ in requests],
                "identities": identities,
            })
        else:
            action = self.path.rsplit("/", 1)[-1]
            self._write({"status": "ok", "state": "offloaded" if action == "offload" else "resident"})

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket = socket.socket(fileno=args.fd)
server.server_address = server.socket.getsockname()
server.serve_forever()
""".strip()
    )
    spec = ManagedScorerProcessSpec(
        process=ManagedProcessConfig(
            python_executable=sys.executable,
            service_root=str(tmp_path),
            startup_timeout=5,
            shutdown_timeout=1,
            log_dir=str(tmp_path / "logs"),
            offline=False,
        ),
        scorer=ManagedScorerConfig(
            name="fake",
            input_kind="image",
            params={"device": "cpu"},
            version="1",
        ),
        client=RemoteRewardSpec(
            base_url="managed://rank-affine",
            required_rewards=("fake",),
            input_kind="image",
            request_batch_size=1,
        ),
    )
    backend = ManagedScorerProcessBackend(config=spec, base_device="cpu")
    process = backend._process
    response = backend.compute_rewards(
        RewardRequest(
            primitives={"text": Texts(texts=["prompt"])},
            generated={"image": Images(pixels=torch.rand(1, 3, 8, 8))},
            sample_ids=["sample-0"],
            group_ids=["group-0"],
        )
    )
    assert response.rewards == [1.0]
    backend.dispose()
    assert process is not None and process.poll() is not None

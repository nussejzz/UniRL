from __future__ import annotations

from types import SimpleNamespace

import pytest

from unirl.trainer.diffusion import DiffusionTrainer, _validate_diffusion_dp_geometry
from unirl.types.sampling import DiffusionSamplingParams


class _Backend:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def offload(self) -> None:
        self.events.append("train.offload")

    def onload(self) -> None:
        self.events.append("train.onload")

    def apply_eval_ema(self) -> None:
        self.events.append("ema.apply")

    def restore_from_eval(self) -> None:
        self.events.append("ema.restore")


class _Rollout:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def wake_up(self) -> None:
        self.events.append("rollout.wake")

    def generate(self, sample):
        self.events.append("rollout.generate")
        if self.fail:
            raise RuntimeError("generate failed")
        return sample

    def sleep(self) -> None:
        self.events.append("rollout.sleep")


class _Sync:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def sync(self) -> None:
        self.events.append("weights.sync")


def _trainer(
    *,
    events: list[str],
    sleep_after_generate: bool = True,
    enable_fsdp_offload: bool = False,
    trainside: bool = False,
    uses_ema: bool = False,
    fail_generate: bool = False,
) -> DiffusionTrainer:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer._layout = "colocate"
    trainer._enable_fsdp_offload = enable_fsdp_offload
    trainer._rollout_sleep_after_generate = sleep_after_generate
    trainer._rollout_is_trainside = trainside
    trainer._uses_ema = uses_ema
    trainer.backend = _Backend(events)
    trainer.rollout = _Rollout(events, fail=fail_generate)
    trainer.weight_sync = _Sync(events)
    return trainer


def test_dp_geometry_distinguishes_prompt_trees_from_generated_rows() -> None:
    _validate_diffusion_dp_geometry(
        batch_size=8,
        samples_per_prompt=8,
        num_updates_per_batch=2,
        rollout_dp_size=8,
        reward_dp_size=8,
        train_dp_size=16,
    )

    with pytest.raises(ValueError, match="root prompt trees"):
        _validate_diffusion_dp_geometry(
            batch_size=8,
            samples_per_prompt=8,
            num_updates_per_batch=2,
            rollout_dp_size=16,
            reward_dp_size=8,
            train_dp_size=16,
        )


def test_dp_geometry_validates_train_update_rows() -> None:
    with pytest.raises(ValueError, match="num_updates_per_batch"):
        _validate_diffusion_dp_geometry(
            batch_size=8,
            samples_per_prompt=6,
            num_updates_per_batch=2,
            rollout_dp_size=8,
            reward_dp_size=8,
            train_dp_size=16,
        )


def test_generate_default_sleep_and_offload_order() -> None:
    events: list[str] = []
    trainer = _trainer(events=events, enable_fsdp_offload=True)
    sample = object()

    assert trainer._generate_for_training(sample, sync_weights=True) is sample
    assert events == [
        "rollout.wake",
        "weights.sync",
        "train.offload",
        "rollout.generate",
        "rollout.sleep",
        "train.onload",
    ]


def test_generate_resident_rollout_skips_sleep() -> None:
    events: list[str] = []
    trainer = _trainer(events=events, sleep_after_generate=False)

    trainer._generate_for_training(object(), sync_weights=False)
    assert events == ["rollout.wake", "rollout.generate"]


def test_generate_failure_restores_rollout_and_train_state() -> None:
    events: list[str] = []
    trainer = _trainer(events=events, enable_fsdp_offload=True, fail_generate=True)

    with pytest.raises(RuntimeError, match="generate failed"):
        trainer._generate_for_training(object(), sync_weights=False)
    assert events == [
        "rollout.wake",
        "train.offload",
        "rollout.generate",
        "rollout.sleep",
        "train.onload",
    ]


def test_trainside_ema_is_restored_after_generate() -> None:
    events: list[str] = []
    trainer = _trainer(events=events, trainside=True, uses_ema=True)

    trainer._generate_for_training(object(), sync_weights=False)
    assert events == [
        "rollout.wake",
        "ema.apply",
        "rollout.generate",
        "ema.restore",
        "rollout.sleep",
    ]


def test_reward_phase_offloads_only_trainside_non_ema() -> None:
    events: list[str] = []
    trainer = _trainer(events=events, enable_fsdp_offload=True, trainside=True)

    with trainer._reward_phase():
        events.append("reward.score")
    assert events == ["train.offload", "reward.score", "train.onload"]


@pytest.mark.parametrize(
    ("configured_sleep", "sleep_after", "expected_sleep"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_evaluate_combines_config_and_callsite_sleep_gates(
    configured_sleep: bool,
    sleep_after: bool,
    expected_sleep: bool,
) -> None:
    events: list[str] = []
    trainer = _trainer(events=events, sleep_after_generate=configured_sleep)
    trainer.sampling_params = {"diffusion": DiffusionSamplingParams()}
    trainer.eval_samples_per_prompt = 1
    trainer.eval_eta = 0.0
    trainer.eval_cfg_text_scale = 1.0
    trainer.eval_num_prompts = 1
    trainer._eval_suites = []
    trainer.data_source = object()
    trainer.reward = object()
    trainer._eval_pass = lambda *args, **kwargs: {"reward": 1.0}
    trainer.wandb_logger = SimpleNamespace(log_eval=lambda *args, **kwargs: None)
    trainer.weight_sync = None

    assert trainer.evaluate(0, sleep_after=sleep_after) == 1.0
    assert ("rollout.sleep" in events) is expected_sleep

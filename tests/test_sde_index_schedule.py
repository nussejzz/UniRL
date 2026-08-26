"""SDE index scheduling tests."""

import pytest

from unirl.sde.index_schedule import AllSDEScheduler, RandomContiguousSDEScheduler, SigmaBandSDEScheduler


def test_sparse_scheduler_is_seeded_and_noncontiguous() -> None:
    scheduler = AllSDEScheduler(
        num_timesteps=24,
        timestep_fraction=(0.0, 10 / 24),
        num_sde_steps=3,
        seed=42,
    )

    first = scheduler.get_sde_indices(0)
    second = scheduler.get_sde_indices(1)

    assert len(first) == len(second) == 3
    assert first <= set(range(10))
    assert second <= set(range(10))
    assert first != second
    assert first == scheduler.get_sde_indices(0)


def test_random_contiguous_scheduler_matches_seeded_window_contract() -> None:
    scheduler = RandomContiguousSDEScheduler(
        num_timesteps=23,
        window_size=3,
        start=0,
        end=23,
        seed=42,
    )

    first = sorted(scheduler.get_sde_indices(0))
    second = sorted(scheduler.get_sde_indices(1))

    assert len(first) == len(second) == 3
    assert first == list(range(first[0], first[0] + 3))
    assert second == list(range(second[0], second[0] + 3))
    assert first != second
    assert first == sorted(scheduler.get_sde_indices(0))


def test_sigma_band_scheduler_covers_visual_noise_bands() -> None:
    scheduler = SigmaBandSDEScheduler(
        num_timesteps=24,
        shift=12.0,
        sigma_bands=((0.85, 0.93), (0.70, 0.84), (0.50, 0.65)),
        seed=42,
    )

    assert scheduler.candidate_pools == (
        (12, 13, 14, 15, 16),
        (17, 18, 19, 20),
        (21, 22),
    )
    first = scheduler.get_sde_indices(0)
    second = scheduler.get_sde_indices(1)
    assert len(first) == len(second) == 3
    assert first == scheduler.get_sde_indices(0)
    assert any(index in scheduler.candidate_pools[0] for index in first)
    assert any(index in scheduler.candidate_pools[1] for index in first)
    assert any(index in scheduler.candidate_pools[2] for index in first)
    assert first != second


def test_sigma_band_scheduler_rejects_overlapping_pools() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        SigmaBandSDEScheduler(
            num_timesteps=24,
            shift=12.0,
            sigma_bands=((0.80, 0.95), (0.70, 0.90)),
        )

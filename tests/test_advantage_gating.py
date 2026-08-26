"""Low-spread group advantage gating tests."""

import torch

from unirl.types.sample import Part


def test_low_spread_group_advantages_are_zeroed() -> None:
    part = Part(
        sample_ids=["prompt:7/0", "prompt:7/1"],
        rewards=torch.tensor([0.78, 0.78000006]),
    )

    result = part.compute_advantages(min_group_std=1e-4)

    torch.testing.assert_close(result.advantages, torch.zeros(2))


def test_informative_group_advantages_are_preserved() -> None:
    part = Part(
        sample_ids=["prompt:7/0", "prompt:7/1"],
        rewards=torch.tensor([0.7, 0.9]),
    )

    result = part.compute_advantages(min_group_std=1e-4)

    assert result.advantages is not None
    assert float(result.advantages.abs().max()) > 0.9

"""MiniMax-H3 sigma-grid and CPS scheduler equivalence tests."""

import math

import pytest
import torch
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from unirl.models.minimax_h3.conditions import MiniMaxH3Conditions
from unirl.models.minimax_h3.diffusion import MiniMaxH3DiffusionStage, _combine_modality_logp
from unirl.sde.kernels import CPSSDEStrategy, CPSSpec
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import compute_trajectory_positions


def _reference_cps(
    sample: torch.Tensor,
    noise_pred: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float,
    prev_sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    std = sigma_next * math.sin(eta * math.pi / 2)
    pred_original = sample - sigma * noise_pred
    noise_estimate = sample + noise_pred * (1 - sigma)
    mean = pred_original * (1 - sigma_next) + noise_estimate * math.sqrt(sigma_next**2 - std**2)
    log_prob = -((prev_sample.detach() - mean) ** 2).mean(dim=tuple(range(1, sample.ndim)))
    return mean, log_prob


def test_h3_video_and_audio_sigma_grids_match_vllm() -> None:
    for shift in (12.0, 3.0):
        unirl_sigmas = get_sigma_schedule(num_steps=24, shift=shift, device=torch.device("cpu"))
        vllm_sigmas = torch.tensor(
            minimax_h3_time_shift_sigmas(num_steps=25, shift_scale=shift),
            dtype=torch.float32,
        )

        assert unirl_sigmas.shape == vllm_sigmas.shape == (25,)
        torch.testing.assert_close(unirl_sigmas, vllm_sigmas, atol=0, rtol=0)
        assert float(unirl_sigmas[0]) == 1.0
        assert float(unirl_sigmas[-1]) == 0.0
        assert torch.all(unirl_sigmas[:-1] > unirl_sigmas[1:])


def test_cps_transition_and_surrogate_logprob_match_reference() -> None:
    torch.manual_seed(9)
    strategy = CPSSDEStrategy()
    sample = torch.randn(2, 3, 4)
    noise_pred = torch.randn_like(sample)
    prev_sample = torch.randn_like(sample)
    sigma = 0.96
    sigma_next = 0.9523809552
    eta = 0.6

    actual_prev, actual_logp, actual_mean = strategy.denoise(
        noise_pred=noise_pred,
        sample=sample,
        sigma=torch.tensor(sigma),
        sigma_next=torch.tensor(sigma_next),
        eta=eta,
        prev_sample=prev_sample,
    )
    expected_mean, expected_logp = _reference_cps(sample, noise_pred, sigma, sigma_next, eta, prev_sample)

    torch.testing.assert_close(actual_prev, prev_sample)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_logp, expected_logp)


def test_cps_eta_zero_is_h3_euler_and_has_no_logprob() -> None:
    torch.manual_seed(17)
    strategy = CPSSDEStrategy()
    sample = torch.randn(1, 7, 5)
    h3_velocity = torch.randn_like(sample)
    sigma = 0.8571428061
    sigma_next = 0.8333333135

    actual, log_prob, _ = strategy.denoise(
        noise_pred=-h3_velocity,
        sample=sample,
        sigma=torch.tensor(sigma),
        sigma_next=torch.tensor(sigma_next),
        eta=0.0,
    )
    expected = sample + (sigma - sigma_next) * h3_velocity

    assert log_prob is None
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_cps_preserves_next_sigma_noise_budget() -> None:
    sigma_next = 0.706
    eta = 0.6
    fresh_noise_coeff = sigma_next * math.sin(eta * math.pi / 2)
    retained_noise_coeff = sigma_next * math.cos(eta * math.pi / 2)

    assert math.isclose(
        fresh_noise_coeff**2 + retained_noise_coeff**2,
        sigma_next**2,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_cps_logprob_gradient_matches_finite_difference() -> None:
    strategy = CPSSDEStrategy()
    sample = torch.tensor([[[0.3, -0.2]]])
    prev_sample = torch.tensor([[[0.8, -0.7]]])
    noise_pred = torch.tensor([[[0.11, -0.09]]], requires_grad=True)
    sigma = torch.tensor(0.8571428)
    sigma_next = torch.tensor(0.8333333)

    _, log_prob, _ = strategy.denoise(
        noise_pred=noise_pred,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=0.6,
        prev_sample=prev_sample,
    )
    assert log_prob is not None
    log_prob.sum().backward()
    autograd_value = float(noise_pred.grad[0, 0, 0])

    def evaluate(value: float) -> float:
        candidate = noise_pred.detach().clone()
        candidate[0, 0, 0] = value
        _, result, _ = strategy.denoise(
            noise_pred=candidate,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=0.6,
            prev_sample=prev_sample,
        )
        assert result is not None
        return float(result.sum())

    center = float(noise_pred.detach()[0, 0, 0])
    epsilon = 1e-3
    finite_difference = (evaluate(center + epsilon) - evaluate(center - epsilon)) / (2 * epsilon)
    assert math.isclose(autograd_value, finite_difference, rel_tol=3e-3, abs_tol=3e-4)


def test_cps_gaussian_logprob_matches_normal_distribution() -> None:
    strategy = CPSSDEStrategy(config=CPSSpec(logprob_mode="gaussian"))
    mean = torch.tensor([[[0.2, -0.1]]])
    sample = torch.tensor([[[0.5, -0.7]]])
    std = torch.tensor(0.3)

    actual = strategy.compute_log_prob(
        prev_sample=sample,
        prev_sample_mean=mean,
        std_var=std,
    )
    expected = torch.distributions.Normal(mean, std).log_prob(sample)

    torch.testing.assert_close(actual, expected)


def test_cps_gaussian_logprob_rejects_zero_variance() -> None:
    strategy = CPSSDEStrategy(config=CPSSpec(logprob_mode="gaussian"))
    with pytest.raises(ValueError, match="positive transition std"):
        strategy.compute_log_prob(
            prev_sample=torch.zeros(1, 2),
            prev_sample_mean=torch.zeros(1, 2),
            std_var=torch.tensor(0.0),
        )


def test_fixed_av_logprob_weights_match_verl_h3_recipe() -> None:
    video = torch.tensor([1.0, 3.0])
    audio = torch.tensor([5.0, 7.0])

    combined = _combine_modality_logp(
        video,
        audio,
        n_video=269,
        n_audio=1,
        video_weight=0.5,
        audio_weight=0.5,
    )

    torch.testing.assert_close(combined, (video + audio) / 2)


def test_h3_stage_trims_padded_rollout_text_embeddings() -> None:
    conditions = MiniMaxH3Conditions(
        text=TextEmbedCondition(
            embeds=torch.randn(1, 7, 4),
            attn_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool),
        )
    )

    trimmed = MiniMaxH3DiffusionStage._trim_padded_text(conditions)

    assert trimmed.text.embeds.shape == (1, 3, 4)
    assert trimmed.text.attn_mask.shape == (1, 3)


def test_sparse_transition_positions_store_only_replay_boundaries() -> None:
    assert compute_trajectory_positions({8, 16, 20}, num_steps=24) == [8, 9, 16, 17, 20, 21]
    assert compute_trajectory_positions({8, 9, 20}, num_steps=24) == [8, 9, 10, 20, 21]

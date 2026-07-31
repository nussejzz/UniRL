from __future__ import annotations

import dataclasses

import pytest
import torch

from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.models.bagel.rl_ops import (
    _encode_vae_posterior_mean,
    activation_checkpoint_bypass_scope,
    update_context_image,
)
from unirl.rollout.engine.vllm_omni.adapters.bagel import BagelInputAdapter, BagelOutputAdapter
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample


def _request(*, image_input: bool, samples_per_prompt: int = 2, eta: float = 0.0) -> Sample:
    root = Part.input(
        ["prompt:0", "prompt:1"],
        primitives={"text": Texts(texts=["make it blue", "add a hat"])},
    )
    parts = [root]
    if image_input:
        parts.append(root.input_child({"image": Images(pixels=torch.rand(2, 3, 16, 16))}))
    params = BagelDiffusionParams(
        samples_per_prompt=samples_per_prompt,
        eta=eta,
        sigmas=torch.linspace(1.0, 0.0, 15),
        sde_indices=(1, 2),
    )
    return Sample.request(*parts).fork(samples_per_prompt, sampling_params=params)


def test_it2i_adapter_keeps_source_images_frontier_aligned() -> None:
    sample = _request(image_input=True)
    input_adapter = BagelInputAdapter("bagel_it2i", image_input=True)
    output_adapter = BagelOutputAdapter("bagel_it2i", image_input=True)

    prompts = input_adapter.build_prompts(sample)
    sampling = input_adapter.build_sampling(sample)[0].kwargs
    conditions = output_adapter.build_conditions(sample, [])["bagel"]

    assert len(prompts) == 4
    assert all(prompt["multi_modal_data"]["image"].mode == "RGB" for prompt in prompts)
    assert sampling["num_outputs_per_prompt"] == 1
    assert "sde_indices" not in sampling["extra_args"]
    assert conditions.batch_size == 4
    assert len(conditions.input_images) == 4
    prompt, source_image, image_shape = conditions.slice(0, 1).single_prompt()
    assert prompt == "make it blue"
    assert source_image is not None
    assert image_shape == (512, 512)


def test_t2i_adapter_preserves_native_prompt_group_packing() -> None:
    sample = _request(image_input=False)
    adapter = BagelInputAdapter("bagel_t2i")

    prompts = adapter.build_prompts(sample)
    sampling = adapter.build_sampling(sample)[0].kwargs

    assert prompts == [{"prompt": "make it blue"}, {"prompt": "add a hat"}]
    assert sampling["num_outputs_per_prompt"] == 2


def test_eta_gate_matches_trainside_deterministic_cutoff() -> None:
    below_cutoff = _request(image_input=True, eta=1e-8)
    at_cutoff = _request(image_input=True, eta=1e-7)
    adapter = BagelInputAdapter("bagel_it2i", image_input=True)

    below = adapter.build_sampling(below_cutoff)[0].kwargs["extra_args"]
    active = adapter.build_sampling(at_cutoff)[0].kwargs["extra_args"]

    assert "sde_indices" not in below
    assert active["sde_indices"] == [1, 2]


def test_explicit_eval_noise_keys_override_lineage_grouping() -> None:
    sample = _request(image_input=True)
    frontier = dataclasses.replace(
        sample.parts[-1],
        init_noise_group_ids=["eval:0", "eval:1", "eval:2", "eval:3"],
    )
    params = dataclasses.replace(
        frontier.sampling_params,
        init_noise_latent_shape=[4, 8],
        init_same_noise=True,
    )
    extra_args = {}

    pack_initial_noise_extra_args(extra_args, frontier, params, caller="test")

    assert extra_args["init_noise_group_ids"] == ["eval:0", "eval:1", "eval:2", "eval:3"]


def test_vae_source_prefill_uses_posterior_mean() -> None:
    class _Reg:
        chunk_dim = 1

    class _VAE:
        reg = _Reg()
        scale_factor = 2.0
        shift_factor = 0.5

        @staticmethod
        def encoder(x: torch.Tensor) -> torch.Tensor:
            mean = x + 1.0
            logvar = torch.full_like(mean, 20.0)
            return torch.cat((mean, logvar), dim=1)

    x = torch.ones(1, 2, 2, 2)
    expected = 2.0 * ((x + 1.0) - 0.5)

    first = _encode_vae_posterior_mean(_VAE(), x)
    second = _encode_vae_posterior_mean(_VAE(), x)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_differentiable_source_prefill_reaches_vae_path() -> None:
    class _Reg:
        chunk_dim = 1

    class _VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.reg = _Reg()
            self.scale_factor = 1.0
            self.shift_factor = 0.0

        def encoder(self, x: torch.Tensor) -> torch.Tensor:
            mean = x * self.weight
            return torch.cat((mean, torch.zeros_like(mean)), dim=1)

    class _Cache:
        def __init__(self, num_layers: int) -> None:
            self.key_cache = {index: None for index in range(num_layers)}
            self.value_cache = {index: None for index in range(num_layers)}

        @property
        def num_layers(self) -> int:
            return len(self.key_cache)

    class _Bagel:
        def __init__(self) -> None:
            self.vae2llm = torch.nn.Linear(1, 1)

        @staticmethod
        def prepare_vae_images(**kwargs):
            image = kwargs["images"][0]
            return {"padded_images": image}, [1], [1]

        @torch.no_grad()
        def forward_cache_update_vae(self, vae_model, past_key_values, padded_images):
            past_key_values.key_cache[0] = vae_model.encode(padded_images)
            return past_key_values

    vae = _VAE()
    bagel = _Bagel()
    bundle = type(
        "_Bundle",
        (),
        {
            "model": bagel,
            "vae": vae,
            "vae_transform": object(),
            "new_token_ids": {},
            "device": "cpu",
        },
    )()
    context = {
        "kv_lens": [0],
        "ropes": [0],
        "past_key_values": _Cache(1),
    }

    updated = update_context_image(
        bundle,
        torch.ones(1, 1, 2, 2),
        context,
        vae=True,
        vit=False,
        differentiable=True,
    )
    updated["past_key_values"].key_cache[0].sum().backward()
    assert vae.weight.grad is not None
    assert vae.weight.grad.item() > 0


def test_differentiable_prefill_bypasses_activation_checkpoint_wrapper() -> None:
    class _Layer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    layer = _Layer()
    original = layer.forward

    def checkpointed(*args, **kwargs):
        raise AssertionError("checkpoint wrapper must be bypassed")

    checkpointed.__wrapped__ = original
    checkpointed._unirl_activation_checkpoint = True
    layer.forward = checkpointed
    language_model = torch.nn.Module()
    language_model.layer = layer
    model = type("_Model", (), {"language_model": language_model})()

    with activation_checkpoint_bypass_scope(model):
        assert torch.equal(layer.forward(torch.tensor(1)), torch.tensor(2))
    assert layer.forward is checkpointed


def test_it2i_and_t2i_validate_opposite_image_contracts() -> None:
    it2i = _request(image_input=True)
    t2i = _request(image_input=False)

    from unirl.rollout.engine.vllm_omni.adapters.bagel import BagelIt2iAdapter, BagelT2iAdapter

    # Validation does not use model/config state, so construct the binders without
    # booting a rollout engine.
    it2i_adapter = BagelIt2iAdapter.__new__(BagelIt2iAdapter)
    it2i_adapter.modality = "bagel_it2i"
    it2i_adapter.image_input = True
    t2i_adapter = BagelT2iAdapter.__new__(BagelT2iAdapter)
    t2i_adapter.modality = "bagel_t2i"
    t2i_adapter.image_input = False

    it2i_adapter.validate_request(it2i)
    t2i_adapter.validate_request(t2i)

    with pytest.raises(ValueError, match="requires image conditioning"):
        it2i_adapter.validate_request(t2i)
    with pytest.raises(ValueError, match="rejects image-bearing"):
        t2i_adapter.validate_request(it2i)


def test_deferred_conditions_reject_missing_prompt() -> None:
    conditions = BagelDiffusionConditions(
        prompts=[None],
        input_images=[None],
        image_shapes=[(512, 512)],
    )

    with pytest.raises(ValueError, match="no prompt present"):
        conditions.single_prompt()

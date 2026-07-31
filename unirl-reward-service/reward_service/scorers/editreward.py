"""EditReward scorer — multi-dimensional reward for instruction-guided image editing.

Wraps the EditRewardInferencer (Qwen2.5-VL-7B based) from the EditReward
package. Evaluates how well an edited image follows the editing instruction
(dim1: instruction following) and visual quality (dim2: visual quality).

Input convention:
    history[0] = (prompt, source_image)   — source image before editing
    history[1] = (prompt, edited_image)   — edited image after editing

The text prompt (editing instruction) is taken from history[0][0].
"""

from __future__ import annotations

import torch
from PIL import Image

from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


class EditRewardScorer(BaseScorer):
    """Multi-head edit reward scorer backed by EditRewardInferencer."""

    name = "editreward"
    sub_metric_names = ("edit_following", "edit_quality")

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        model_name_or_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rm_head_type: str = "ranknet_multi_head",
        offload_between_calls: bool = False,
    ) -> None:
        import os

        from reward_service.scorers._editreward import EditRewardInferencer

        self._target_device = device if torch.cuda.is_available() else "cpu"
        self._offload_between_calls = bool(offload_between_calls and self._target_device != "cpu")
        initial_device = "cpu" if self._offload_between_calls else self._target_device
        self._rm_head_type = rm_head_type

        # If checkpoint_path looks like a HF repo ID (not a local dir),
        # download it via huggingface_hub first.
        if not os.path.isdir(checkpoint_path) and "/" in checkpoint_path:
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(repo_id=checkpoint_path)

        self.inferencer = EditRewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            device=initial_device,
            reward_dim="dim1",
            rm_head_type=rm_head_type,
        )

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        try:
            if self._offload_between_calls:
                self._move_to(self._target_device)
            return self._score_batch(items)
        finally:
            if self._offload_between_calls:
                self._move_to("cpu")
                torch.cuda.empty_cache()

    def _score_batch(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        """Score valid items in one forward, falling back per item on failure."""
        nan_result = {name: float("nan") for name in self.sub_metric_names}
        results = [dict(nan_result) for _ in items]
        rows: list[int] = []
        prompts: list[str] = []
        source_images: list[Image.Image] = []
        edited_images: list[Image.Image] = []
        for i, item in enumerate(items):
            try:
                prompt, source_image, edited_image = self._unpack_item(item)
                rows.append(i)
                prompts.append(prompt)
                source_images.append(source_image)
                edited_images.append(edited_image)
            except (TypeError, ValueError):
                continue

        if rows:
            try:
                rewards = self.inferencer.reward(
                    prompts=prompts,
                    image_src=source_images,
                    image_paths=edited_images,
                )
                for row, i in enumerate(rows):
                    results[i] = self._shape_reward(rewards, row)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception:
                # A single bad item must not nan the whole batch — fall back to
                # the one-at-a-time path only on the error case.
                for i in rows:
                    try:
                        results[i] = self._score_single(items[i])
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception:
                        results[i] = dict(nan_result)

        return results

    def _move_to(self, device: str) -> None:
        """Move the 7B scorer as one unit and keep input placement in sync.

        In time-multiplexed deployments the scorer stays on CPU outside scoring,
        moves to its assigned CUDA device for one call, then returns to CPU.
        """
        self.inferencer.model.to(device)
        self.inferencer.device = device

    @staticmethod
    def _unpack_item(item: ScoreItem) -> tuple[str, Image.Image, Image.Image]:
        """Validate and unpack one source/edit pair."""
        if len(item.history) < 2:
            raise ValueError(f"EditReward requires 2 history turns (source + edited), got {len(item.history)}")
        prompt, source_image = item.history[0]
        _, edited_image = item.history[1]
        if source_image is None or edited_image is None:
            raise ValueError("Both source and edited images must be provided")
        return prompt, source_image, edited_image

    def _shape_reward(self, rewards, row: int) -> dict[str, float]:
        """Map row ``row`` of a batched ``reward()`` output to the sub-metric dict.

        Mirrors :meth:`_score_single`'s shaping exactly, indexed by batch row, so
        the single batched forward produces the same per-item values the
        one-at-a-time path produced.
        """
        if self._rm_head_type == "ranknet_multi_head":
            if isinstance(rewards, torch.Tensor):
                if rewards.dim() >= 2 and rewards.shape[-1] >= 2:
                    return {
                        "edit_following": float(rewards[row, 0].item()),
                        "edit_quality": float(rewards[row, 1].item()),
                    }
                return {
                    "edit_following": float(rewards[row].item()),
                    "edit_quality": float("nan"),
                }
            if isinstance(rewards, (list, tuple)):
                scores = [float(r[row].item()) if torch.is_tensor(r) else float(r) for r in rewards]
                return {
                    "edit_following": scores[0] if len(scores) > 0 else float("nan"),
                    "edit_quality": scores[1] if len(scores) > 1 else float("nan"),
                }
        else:
            if isinstance(rewards, torch.Tensor):
                val = float(rewards[row, 0].item()) if rewards.dim() >= 2 else float(rewards[row].item())
            else:
                val = float(rewards[row])
            return {
                "edit_following": val,
                "edit_quality": float("nan"),
            }
        return {k: float("nan") for k in self.sub_metric_names}

    def _score_single(self, item: ScoreItem) -> dict[str, float]:
        """Score a single item.

        Expects item.history to have at least 2 turns:
            history[0] = (prompt, source_image)
            history[1] = (prompt, edited_image)
        """
        prompt, source_image, edited_image = self._unpack_item(item)

        # EditRewardInferencer.reward() accepts paths or PIL images
        # (process_vision_info handles PIL.Image directly)
        rewards = self.inferencer.reward(
            prompts=[prompt],
            image_src=[source_image],
            image_paths=[edited_image],
        )
        return self._shape_reward(rewards, row=0)

    def close(self) -> None:
        if hasattr(self, "inferencer"):
            del self.inferencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register("editreward", EditRewardScorer)

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

_DEBUG_KEY_STR = "__openpi_debug__"
_DEBUG_KEY_BYTES = b"__openpi_debug__"


def _to_numpy_leaf(x: Any) -> Any:
    """Best-effort conversion of array-like leaves to numpy for msgpack serialization."""
    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        return x
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # JAX arrays + array-like objects.
    try:
        return np.asarray(x)
    except Exception:
        return x


def _masked_mean(x: Any, mask: Any) -> Any:
    """Compute mean over sequence dimension (axis=1) using a boolean mask (same framework as x)."""
    # x: (b, s, d), mask: (b, s)
    if isinstance(x, torch.Tensor):
        mask_f = mask.to(dtype=x.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom
    # JAX / numpy
    mask_f = mask.astype(x.dtype)
    denom = jnp.clip(jnp.sum(mask_f, axis=1, keepdims=True), a_min=1.0)
    return jnp.sum(x * mask_f[..., None], axis=1) / denom


def _summarize_vec(vec_1d: Any, *, head_dim: int, include_full: bool) -> dict[str, Any]:
    """Return small, print-friendly summary of a 1D embedding vector."""
    if isinstance(vec_1d, torch.Tensor):
        v = vec_1d.to(dtype=torch.float32)
        out: dict[str, Any] = {
            "norm": torch.linalg.vector_norm(v).item(),
            "head": v[:head_dim].detach().cpu().numpy(),
        }
        if include_full:
            out["full"] = v.detach().cpu().numpy()
        return out

    v = jnp.asarray(vec_1d, dtype=jnp.float32)
    out = {
        "norm": float(jnp.linalg.norm(v)),
        "head": np.asarray(v[:head_dim]),
    }
    if include_full:
        out["full"] = np.asarray(v)
    return out


def _is_truthy_flag(debug_req: Any, key: str) -> bool:
    if debug_req is True:
        return True
    if isinstance(debug_req, dict):
        return bool(debug_req.get(key, False))
    return False


def _debug_int(debug_req: Any, key: str, default: int) -> int:
    if isinstance(debug_req, dict):
        try:
            return int(debug_req.get(key, default))
        except Exception:
            return default
    return default


def _debug_bool(debug_req: Any, key: str, default: bool) -> bool:
    if isinstance(debug_req, dict):
        try:
            return bool(debug_req.get(key, default))
        except Exception:
            return default
    return default


def _compute_vlm_debug(
    *,
    model: Any,
    observation: _model.Observation,
    debug_req: Any,
) -> dict[str, Any] | None:
    """Compute a lightweight summary of the vision-language (prefix) expert outputs for pi0/pi05 models.

    This is intended for debugging/inspection and should be requested explicitly by the client.
    """
    if not _is_truthy_flag(debug_req, "vlm"):
        return None

    # Keep payload small by default.
    head_dim = max(1, _debug_int(debug_req, "vlm_head_dim", 16))
    include_full = _debug_bool(debug_req, "vlm_full", False)
    include_images = _debug_bool(debug_req, "vlm_images", True)
    include_prompt = _debug_bool(debug_req, "vlm_prompt", True)

    # We only support pi0/pi05 models (they have embed_prefix + PaliGemma).
    if not (hasattr(model, "embed_prefix") and hasattr(model, "PaliGemma")):
        return {
            "error": "VLM debug is only supported for pi0/pi05-style models (missing embed_prefix/PaliGemma).",
        }

    try:
        # JAX path.
        if not isinstance(observation.state, torch.Tensor):
            from openpi.models.pi0 import make_attn_mask as _make_attn_mask  # local import to avoid cycles

            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            prefix_attn_mask = _make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

            prefix_pooled = _masked_mean(prefix_out, prefix_mask)[0]
            vlm: dict[str, Any] = {
                "pi05": bool(getattr(model, "pi05", False)),
                "head_dim": head_dim,
                "prefix_tokens": int(np.asarray(prefix_mask[0]).sum()),
                "prefix": _summarize_vec(prefix_pooled, head_dim=head_dim, include_full=include_full),
            }

            if include_prompt and observation.tokenized_prompt_mask is not None:
                prompt_len = int(observation.tokenized_prompt_mask.shape[1])
                prompt_out = prefix_out[:, -prompt_len:, :]
                prompt_pooled = _masked_mean(prompt_out, observation.tokenized_prompt_mask)[0]
                vlm["prompt_tokens"] = int(np.asarray(observation.tokenized_prompt_mask[0]).sum())
                vlm["prompt"] = _summarize_vec(prompt_pooled, head_dim=head_dim, include_full=include_full)

            if include_images and observation.images:
                prompt_len = int(observation.tokenized_prompt_mask.shape[1]) if observation.tokenized_prompt_mask is not None else 0
                prefix_len = int(prefix_out.shape[1])
                image_total_len = prefix_len - prompt_len
                num_images = len(observation.images)
                if num_images > 0 and image_total_len % num_images == 0:
                    per_image_len = image_total_len // num_images
                    image_names = list(observation.images.keys())
                    images_out: dict[str, Any] = {}
                    for i, name in enumerate(image_names):
                        s = slice(i * per_image_len, (i + 1) * per_image_len)
                        img_pooled = _masked_mean(prefix_out[:, s, :], prefix_mask[:, s])[0]
                        images_out[name] = _summarize_vec(img_pooled, head_dim=head_dim, include_full=include_full)
                        # also expose whether that view was masked out
                        if observation.image_masks and name in observation.image_masks:
                            images_out[name]["mask"] = bool(np.asarray(observation.image_masks[name][0]))
                    vlm["image_tokens_per_view"] = int(per_image_len)
                    vlm["images"] = images_out
                else:
                    vlm["images"] = {"error": "Could not infer per-view token boundaries for image tokens."}

            return vlm

        # PyTorch path.
        # We re-run a prefix-only forward to obtain prefix hidden states (debug-only).
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks as _make_att_2d_masks  # noqa: WPS433

        # PI0Pytorch has helpers for preprocessing + masks.
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)  # noqa: SLF001
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001

        (prefix_out, _), _ = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        prefix_pooled = _masked_mean(prefix_out, prefix_pad_masks)[0]
        vlm_t: dict[str, Any] = {
            "pi05": bool(getattr(model, "pi05", False)),
            "head_dim": head_dim,
            "prefix_tokens": int(prefix_pad_masks[0].sum().item()),
            "prefix": _summarize_vec(prefix_pooled, head_dim=head_dim, include_full=include_full),
        }

        if include_prompt:
            prompt_len = int(lang_masks.shape[1])
            prompt_out = prefix_out[:, -prompt_len:, :]
            prompt_pooled = _masked_mean(prompt_out, lang_masks)[0]
            vlm_t["prompt_tokens"] = int(lang_masks[0].sum().item())
            vlm_t["prompt"] = _summarize_vec(prompt_pooled, head_dim=head_dim, include_full=include_full)

        if include_images:
            prompt_len = int(lang_masks.shape[1])
            prefix_len = int(prefix_out.shape[1])
            image_total_len = prefix_len - prompt_len
            num_images = len(images)
            if num_images > 0 and image_total_len % num_images == 0:
                per_image_len = image_total_len // num_images
                # We don't have stable view names in the pytorch preprocessing; use indices.
                images_out_t: dict[str, Any] = {}
                for i in range(num_images):
                    s = slice(i * per_image_len, (i + 1) * per_image_len)
                    img_pooled = _masked_mean(prefix_out[:, s, :], prefix_pad_masks[:, s])[0]
                    images_out_t[f"image_{i}"] = _summarize_vec(img_pooled, head_dim=head_dim, include_full=include_full)
                    images_out_t[f"image_{i}"]["mask"] = bool(prefix_pad_masks[0, s].any().item())
                vlm_t["image_tokens_per_view"] = int(per_image_len)
                vlm_t["images"] = images_out_t
            else:
                vlm_t["images"] = {"error": "Could not infer per-view token boundaries for image tokens."}

        return vlm_t

    except Exception as e:  # pragma: no cover - debug-only path
        return {"error": f"Failed to compute VLM debug: {e!s}"}


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Optional debug request embedded in the raw observation dict (before transforms).
        debug_req = None
        if isinstance(obs, dict):
            if _DEBUG_KEY_STR in obs:
                debug_req = obs.pop(_DEBUG_KEY_STR)
            elif _DEBUG_KEY_BYTES in obs:
                debug_req = obs.pop(_DEBUG_KEY_BYTES)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        vlm_debug = _compute_vlm_debug(model=self._model, observation=observation, debug_req=debug_req)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if vlm_debug is not None:
            outputs["vlm"] = jax.tree.map(_to_numpy_leaf, vlm_debug)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

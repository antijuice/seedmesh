"""``WrappedDeepseekV3Block`` -- one hosted DeepSeek-V3 / Kimi-K2 layer.

Same strategy as the Llama and Mixtral ports: delegate to the stock decoder layer, and own
only the four things a block-hosting server needs that no ``DeepseekV3Model`` is there to
provide. Two of those are shared with Llama and imported rather than copied
(``build_causal_mask``, ``_SingleLayerCache``); the third, rotary ownership, is identical in
shape; the fourth -- cache translation -- is **not**, and that is the whole content of this
file.

Multi-head latent attention stores K and V at *different* widths. Measured directly
(spike/moe/probe_mla_shapes.py) on a tiny config with qk_nope=16, qk_rope=8, v_head_dim=16::

    key   (1, 4, 6, 24)      <- qk_nope + qk_rope
    value (1, 4, 6, 16)      <- v_head_dim

transformers decompresses the latent back to full K/V *before* ``Cache.update``, and the
``F.pad`` that widens V to match K happens after retrieval, not before storage. The Llama
translation does ``value_states.view(*key_states.shape)``, which silently assumes one width
for both -- correct for MHA and GQA, wrong here. An earlier note in docs/findings-moe.md
claimed the translation "transfers unchanged"; measuring it before writing this file is what
caught that.

Getting it wrong would not raise. It would reshape one tensor into another's dimensions and
quietly corrupt attention -- and in a swarm that judges servers by whether their outputs
agree, silent corruption is indistinguishable from cheating.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Config,
    DeepseekV3DecoderLayer,
    DeepseekV3RotaryEmbedding,
)

# Architecture-independent, already exercised by the Llama port's equivalence tests.
from petals.models.llama.block import _SingleLayerCache, build_causal_mask


class WrappedDeepseekV3Block(DeepseekV3DecoderLayer):
    """A DeepSeek-V3 layer, self-sufficient enough to run outside a ``DeepseekV3Model``.

    Subclasses rather than wraps so parameter names are unchanged and block checkpoints load
    unmodified. Note that layers are **not** interchangeable in this architecture: with
    ``first_k_dense_replace = 3`` the first three are dense MLPs and the rest are MoE, a 19x
    size difference. ``layer_idx`` therefore genuinely matters here.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int = 0,
        attn_implementation: str = "eager",
        experts_implementation: str = "grouped_mm",
    ) -> None:
        # Deterministic kernels. Verification compares two servers' outputs against a
        # calibrated tolerance, so an unpinned kernel means unpredictable numerical noise.
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = attn_implementation
        # And for experts this is correctness, not just determinism: measured on Mixtral,
        # leaving it unset returns NaN. See docs/findings-moe.md.
        if getattr(config, "_experts_implementation", None) is None:
            config._experts_implementation = experts_implementation

        super().__init__(config, layer_idx)
        # No DeepseekV3Model here to supply position embeddings.
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

        # Cached because the forward path needs both widths on every call.
        self._key_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self._value_dim = config.v_head_dim
        self._n_heads = config.num_attention_heads

    # ---- cache translation -----------------------------------------------------
    #
    # Unlike Llama's, these carry key_dim and value_dim separately. MLA is the reason.

    def _cache_from_petals(
        self, layer_past: tuple[torch.Tensor, torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Petals' flattened cache -> ``(batch, heads, seq, dim)`` for K and V."""
        key_states, value_states = layer_past
        seq_length = key_states.shape[-1]
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, self._n_heads, seq_length, self._key_dim)
        value_states = value_states.reshape(
            batch_size, self._n_heads, seq_length, self._value_dim
        )
        return key_states, value_states

    def _cache_to_petals(
        self, key_states: torch.Tensor, value_states: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, heads, seq, dim)`` -> Petals' flattened cache."""
        seq_length = key_states.shape[2]
        flat_keys = key_states.reshape(
            batch_size * self._n_heads, seq_length, self._key_dim
        ).permute(0, 2, 1)
        flat_values = value_states.reshape(
            batch_size * self._n_heads, seq_length, self._value_dim
        )
        return flat_keys, flat_values

    # ---- forward ---------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple:
        """Run this block, keeping Petals' external calling convention.

        Returns ``(hidden_states,)`` or ``(hidden_states, layer_past)`` when ``use_cache``.
        """
        batch_size, query_length, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        past_key, past_value = None, None
        past_length = 0
        if layer_past is not None:
            past_key, past_value = self._cache_from_petals(layer_past, batch_size)
            past_length = past_key.shape[2]

        key_length = past_length + query_length

        if position_ids is None:
            position_ids = torch.arange(
                past_length, key_length, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = build_causal_mask(
            batch_size, query_length, key_length, dtype, device, padding_mask=attention_mask
        )

        cache = _SingleLayerCache(past_key, past_value) if use_cache or layer_past else None

        outputs = super().forward(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        if not use_cache:
            return (hidden_states,)

        present = self._cache_to_petals(cache.key_cache, cache.value_cache, batch_size)
        return (hidden_states, present)

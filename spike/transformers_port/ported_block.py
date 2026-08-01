"""A port of Petals' ``WrappedLlamaBlock`` to modern transformers.

The porting strategy is the finding, so it is worth stating up front.

Petals' original file is ~300 lines across three classes. Most of it is not doing what a
distributed inference server actually needs -- it is a hand-copied fork of transformers'
attention implementation, carried so that two CUDA-graph micro-optimizations could be
spliced in (graphed rotary application and graphed RMSNorm, both of which only fire for
single-token inference on CUDA). That copied attention body is precisely the part that
tracks transformers' internals, and therefore precisely the part that broke.

So this port **deletes it** rather than updating it. What a block-hosting server genuinely
needs from Petals' wrapper is four things, none of which require forking attention:

1. own a rotary embedding, because a server hosts a slice of layers with no ``LlamaModel``
   around it to supply position embeddings;
2. build its own causal mask, because there is no model-level mask plumbing either;
3. translate between Petals' cache layout and whatever transformers currently wants;
4. keep parameter names identical so existing checkpoints still load.

Everything else delegates to the stock ``LlamaDecoderLayer``, which HuggingFace maintains.

The trade: the CUDA-graph optimizations are gone. They were worth something for
single-token decode on CUDA. They are not worth owning a fork of attention that breaks on
every transformers release -- and a volunteer swarm's bottleneck is network latency between
hops, not microseconds of kernel launch overhead.

Compatibility is asserted numerically by ``test_equivalence.py``, not assumed.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)


def build_causal_mask(
    batch_size: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Additive 4-D causal mask of shape ``(batch, 1, query_length, key_length)``.

    Replaces ``transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask``,
    which was private, is gone, and was the single most brittle import in the original file.

    Written out in plain torch on purpose. It is about ten lines, it depends on no
    transformers API at all, and so it cannot break again on the next release. Trading a
    private-API dependency for a small amount of owned code is the right deal whenever the
    code is this simple.
    """
    minimum = torch.finfo(dtype).min

    # Query position i (absolute: key_length - query_length + i) may attend to key j <= i.
    offset = key_length - query_length
    query_positions = torch.arange(query_length, device=device).unsqueeze(-1) + offset
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    causal = key_positions > query_positions  # True where attention is forbidden

    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    mask.masked_fill_(causal, minimum)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, key_length)

    if padding_mask is not None:
        # padding_mask is (batch, key_length), True/1 where the token is real.
        pad = padding_mask[:, None, None, :].to(torch.bool)
        mask = mask.masked_fill(~pad, minimum)

    return mask.contiguous()


class WrappedLlamaBlock(LlamaDecoderLayer):
    """One hosted Llama block, self-sufficient enough to run outside a ``LlamaModel``.

    Subclasses rather than wraps ``LlamaDecoderLayer`` so parameter names are unchanged and
    existing Petals block checkpoints keep loading. ``LlamaRotaryEmbedding`` contributes
    only a non-persistent ``inv_freq`` buffer, so it does not enter ``state_dict``.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int = 0,
        attn_implementation: str = "eager",
    ) -> None:
        # Attention dispatch is normally set by the model wrapper. Hosting a bare layer
        # leaves it None, and transformers warns and falls back. Set it explicitly so the
        # kernel choice is deterministic -- it must be, because verification compares two
        # servers' outputs and an unpredictable kernel means unpredictable numerical noise.
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = attn_implementation

        super().__init__(config, layer_idx)
        # A server hosts a slice of layers with no model wrapper to supply position
        # embeddings, so the block owns its own rotary embedding.
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    # ---- cache translation -----------------------------------------------------

    @staticmethod
    def _cache_from_petals(
        layer_past: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Petals' flattened (bloom-style) cache -> ``(batch, heads, seq, head_dim)``."""
        key_states, value_states = layer_past
        seq_length = key_states.shape[-1]
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(batch_size, num_kv_heads, seq_length, head_dim)
        value_states = value_states.view(*key_states.shape)
        return key_states, value_states

    @staticmethod
    def _cache_to_petals(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, heads, seq, head_dim)`` -> Petals' flattened (bloom-style) cache."""
        seq_length = key_states.shape[2]
        value_states = value_states.reshape(batch_size * num_kv_heads, seq_length, head_dim)
        key_states = key_states.reshape(*value_states.shape).permute(0, 2, 1)
        return key_states, value_states

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
        num_kv_heads = self.self_attn.config.num_key_value_heads
        head_dim = self.self_attn.head_dim
        device, dtype = hidden_states.device, hidden_states.dtype

        past_key, past_value = None, None
        past_length = 0
        if layer_past is not None:
            past_key, past_value = self._cache_from_petals(
                layer_past, batch_size, num_kv_heads, head_dim
            )
            past_length = past_key.shape[2]

        key_length = past_length + query_length

        if position_ids is None:
            position_ids = torch.arange(
                past_length, key_length, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Position embeddings moved out of LlamaAttention and up to LlamaModel in
        # transformers 4.48. Since there is no LlamaModel here, the block computes them.
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
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        if not use_cache:
            return (hidden_states,)

        present = self._cache_to_petals(
            cache.key_cache, cache.value_cache, batch_size, num_kv_heads, head_dim
        )
        return (hidden_states, present)


class _SingleLayerCache(nn.Module):
    """Minimal ``Cache``-compatible object holding exactly one layer.

    Petals hosts individual blocks with externally managed attention caches, so
    transformers' multi-layer ``DynamicCache`` (indexed by ``layer_idx``, and increasingly
    entangled with model-level cache management) is the wrong shape. This implements only
    the ``update`` contract an attention module actually calls.

    Deliberately small: the narrower the surface touched, the less there is to re-port.
    """

    def __init__(
        self,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None:
            self.key_cache, self.value_cache = key_states, value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)
        return self.key_cache, self.value_cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if self.key_cache is None else self.key_cache.shape[2]

    def __len__(self) -> int:
        return 1

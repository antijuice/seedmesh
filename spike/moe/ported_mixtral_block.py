"""A port of Petals' ``WrappedMixtralBlock`` to modern transformers.

Same strategy as the Llama port, and for the same reason: delete what tracks transformers'
internals, keep only what a block-hosting server genuinely needs. Petals' original Mixtral
wrapper is 113 lines that pass ``position_ids`` but never build ``position_embeddings`` --
and since transformers 4.48 moved RoPE out of the attention layer and up into the model,
that lands as::

    cos, sin = position_embeddings
    TypeError: cannot unpack non-iterable NoneType object

on the very first forward pass. Mixtral matters because it is the only mixture-of-experts
architecture Petals registers, and therefore the whole basis for hosting an MoE at all.

Three of the four things this needs are architecture-independent and already written and
tested for Llama, so they are imported rather than copied:

  * ``build_causal_mask`` -- replaces the private, now-removed
    ``_prepare_4d_causal_attention_mask`` the original called
  * ``_SingleLayerCache`` -- Petals hosts one block with an externally managed cache, which
    is not the shape of transformers' multi-layer ``DynamicCache``
  * the Petals cache-layout translation -- mixture-of-experts changes the MLP, not attention,
    so the KV cache layout is identical to Llama's

What is genuinely Mixtral-specific is one line: it owns a ``MixtralRotaryEmbedding``.

Expert routing happens entirely inside ``MixtralSparseMoeBlock``, below this wrapper. That is
why an MoE costs no more on the wire than a dense model of the same hidden size, and why this
port is small. See docs/findings-moe.md.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralRotaryEmbedding,
)

# Architecture-independent, and already exercised by the Llama port's equivalence tests.
# Importing rather than duplicating so the two cannot drift apart -- a subtly different
# causal mask between two block types would surface as honest servers disagreeing, which
# this project's verification layer would read as fraud.
from petals.models.llama.block import _SingleLayerCache, build_causal_mask


class WrappedMixtralBlock(MixtralDecoderLayer):
    """One hosted Mixtral block, self-sufficient enough to run outside a ``MixtralModel``.

    Subclasses rather than wraps so parameter names are unchanged and existing Petals block
    checkpoints keep loading. ``MixtralRotaryEmbedding`` contributes only a non-persistent
    ``inv_freq`` buffer, so it does not enter ``state_dict``.
    """

    def __init__(
        self,
        config: MixtralConfig,
        layer_idx: int = 0,
        attn_implementation: str = "eager",
        experts_implementation: str = "grouped_mm",
    ) -> None:
        # Deterministic kernel choice, for the same reason as the Llama port: verification
        # compares two servers' outputs, so an unpredictable attention implementation means
        # unpredictable numerical noise and a tolerance table that means nothing.
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = attn_implementation

        # The same dispatch exists for mixture-of-experts in transformers 5.x, and leaving it
        # unset is not merely non-deterministic -- it is broken. Measured on CPU/fp32 with a
        # tiny Mixtral config:
        #
        #     None        NaN
        #     batched_mm  NaN
        #     grouped_mm  finite            <- the only one that works
        #     deepgemm    requires bfloat16
        #     sonicmoe    requires CUDA
        #
        # A hosted block has no model wrapper to set this, so without it every Mixtral server
        # silently returns NaN. Note this belongs in ComputeProfile beside `attn`: two honest
        # servers on different expert kernels will disagree numerically, and this project's
        # verification layer reads disagreement as fraud.
        if getattr(config, "_experts_implementation", None) is None:
            config._experts_implementation = experts_implementation

        # Sliding-window attention would need a mask this block does not build. Mixtral-8x7B
        # ships `sliding_window: null` so the common path is unaffected -- but silently
        # ignoring a configured window would produce quietly wrong attention, and quietly
        # wrong output is the worst possible failure in a swarm that judges servers by
        # whether their outputs agree. Refuse instead.
        window = getattr(config, "sliding_window", None)
        if window is not None:
            raise NotImplementedError(
                f"sliding_window={window} is set, but this block builds a dense causal mask. "
                "Hosting it would silently produce wrong attention. See "
                "docs/findings-moe.md."
            )

        super().__init__(config, layer_idx)
        # A server hosts a slice of layers with no model wrapper to supply position
        # embeddings, so the block owns its own rotary embedding.
        self.rotary_emb = MixtralRotaryEmbedding(config=config)

    # ---- cache translation -----------------------------------------------------
    #
    # Identical to Llama's: MoE replaces the MLP, not attention, so the KV cache is
    # unchanged in both shape and layout.

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

        # The whole reason this port exists: no MixtralModel here to compute these.
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

        # transformers 5.x returns a bare tensor here; older versions returned a tuple, and
        # with output_router_logits=True there is a second element. Take the hidden states
        # either way.
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        if not use_cache:
            return (hidden_states,)

        present = self._cache_to_petals(
            cache.key_cache, cache.value_cache, batch_size, num_kv_heads, head_dim
        )
        return (hidden_states, present)

"""Petals' quantization logic, reproduced as-is, to see whether it still works.

The whole of Petals' bitsandbytes coupling is four symbols, all in
``petals/utils/convert_block.py::quantize_module``:

    bnb.nn.Linear8bitLt(in, out, bias, has_fp16_weights=False, threshold=6.0)
    bnb.nn.Int8Params(data, requires_grad=False, has_fp16_weights=False)
    bnb.nn.LinearNF4(in, out, bias, compress_statistics=True)
    bnb.nn.Params4bit(data, requires_grad=False, quant_type="nf4", blocksize=64,
                      compress_statistics=True)

That is a far narrower surface than the transformers coupling, which forked an entire
attention implementation. The interesting question is therefore not "how much code is
there" but "did these four constructors keep their signatures across nine minor versions"
(Petals pins 0.41.1; current is 0.50.0).

``quantize_module_petals`` below is the upstream logic, deliberately unmodified, so any
failure is a real API break rather than a transcription artefact.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SKIP_MODULES = ("lm_head", "score")


def quantize_module_petals(model: nn.Module, *, quant_type: str) -> nn.Module:
    """Verbatim reproduction of Petals' ``quantize_module`` (MIT licensed).

    Kept byte-for-byte in behaviour so the spike measures upstream compatibility rather
    than a rewrite. ``quant_type`` is "int8" or "nf4".
    """
    import bitsandbytes as bnb

    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            quantize_module_petals(module, quant_type=quant_type)

        if isinstance(module, torch.nn.Linear) and n not in SKIP_MODULES:
            assert module.weight.device.type == "cpu", (
                f"expected linear layers on CPU, got {module.weight.device}"
            )
            if quant_type == "int8":
                model._modules[n] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,  # Default from the LLM.int8() paper
                )
                model._modules[n].weight = bnb.nn.Int8Params(
                    module.weight.data, requires_grad=False, has_fp16_weights=False
                ).to(module.weight.dtype)
            elif quant_type == "nf4":
                compress_statistics = True
                model._modules[n] = bnb.nn.LinearNF4(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compress_statistics=compress_statistics,
                )
                model._modules[n].weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                    blocksize=64,
                    compress_statistics=compress_statistics,
                ).to(module.weight.dtype)
            else:
                raise ValueError(f"Unsupported quant_type='{quant_type}'")
            model._modules[n].bias = module.bias
    return model


def measure_module_bytes(model: nn.Module) -> int:
    """Actual bytes held by parameters and buffers, counting real element sizes.

    Quantized parameters report exotic dtypes (``int8``, packed ``uint8`` for NF4), so
    ``numel() * 2`` style estimates are wrong. This walks the real storage, which is what a
    volunteer's VRAM actually sees -- and what the onboarding wizard has to predict before
    it can recommend a block count.
    """
    total = 0
    seen: set[int] = set()

    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        storage_id = tensor.data_ptr()
        if storage_id in seen:
            continue
        seen.add(storage_id)
        total += tensor.numel() * tensor.element_size()

        # NF4 keeps its dequantization statistics off to the side, in quant_state rather
        # than in parameters(). Missing them understates the real footprint.
        state = getattr(tensor, "quant_state", None)
        if state is not None:
            for attribute in ("absmax", "code", "offset", "state2"):
                value = getattr(state, attribute, None)
                if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
                    seen.add(value.data_ptr())
                    total += value.numel() * value.element_size()
                elif value is not None and hasattr(value, "absmax"):
                    inner = value.absmax
                    if isinstance(inner, torch.Tensor) and inner.data_ptr() not in seen:
                        seen.add(inner.data_ptr())
                        total += inner.numel() * inner.element_size()

    return total

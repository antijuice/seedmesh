"""Distributed config for DeepSeek-V3 and the models that share its implementation."""

import os
from typing import Optional, Union

from hivemind.utils import get_logger
from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.deepseek_v3.block import WrappedDeepseekV3Block

logger = get_logger(__name__)


class DistributedDeepseekV3Config(DeepseekV3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedDeepseekV3Block
    attn_class = DeepseekV3Attention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        # Multi-head latent attention shares nothing between heads: every head keeps its own
        # K/V, so there is exactly one group. Petals divides its attention-cache budget by
        # this, and returning the Llama-style ratio here would under-reserve.
        return 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args,
        dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).split("/")[-1].replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        # use_cache=False gives identical results but is slower and unsupported by Petals.
        config.use_cache = True
        return result

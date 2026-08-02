"""DeepSeek-V3 with every transformer layer hosted by the swarm.

Structurally identical to the Llama version -- the distributed plumbing does not care what
is inside a block. Kept as its own module rather than parameterised because Petals'
registry maps one config class to one model class.
"""

from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
)

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.deepseek_v3.config import DistributedDeepseekV3Config

logger = get_logger(__name__)


class DistributedDeepseekV3Model(FromPretrainedMixin, PTuneMixin, DeepseekV3Model):
    """DeepseekV3Model, but all transformer layers are hosted by the swarm."""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedDeepseekV3Config

    def __init__(self, config: DistributedDeepseekV3Config, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = (
            self.config.tuning_mode
            and "ptune" in self.config.tuning_mode
            and self.layers.position == 0
        )
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            intermediate_prompts = None

        hidden_states = self.layers(
            inputs_embeds, prompts=intermediate_prompts, hypo_ids=None
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


class DistributedDeepseekV3ForCausalLM(
    FromPretrainedMixin, RemoteGenerationMixin, DeepseekV3ForCausalLM
):
    _keys_to_ignore_on_load_missing = DistributedDeepseekV3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = (
        DistributedDeepseekV3Model._keys_to_ignore_on_load_unexpected
    )

    config_class = DistributedDeepseekV3Config

    def __init__(self, config: DistributedDeepseekV3Config):
        DeepseekV3PreTrainedModel.__init__(self, config)
        self.model = DistributedDeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedDeepseekV3Model:  # For compatibility
        return self.model

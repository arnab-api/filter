import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import (
    PatchSpec,
    get_module_nnsight,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class AttentionInformation(DataClassJsonMixin):
    tokenized_prompt: list[str]
    attention_matrices: np.ndarray
    logits: torch.Tensor | None = None

    def _init__(
        self, prompt: str, tokenized_prompt: list[str], attention_matrices: torch.tensor
    ):
        assert len(tokenized_prompt) == attention_matrices.shape[-1], (
            "Tokenized prompt and attention matrices must have the same length"
        )
        assert len(attention_matrices.shape) == 4, (
            "Attention matrices must be of shape (layers, heads, tokens, tokens)"
        )
        assert attention_matrices.shape[-1] == attention_matrices.shape[-2], (
            "Attention matrices must be square"
        )

        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt
        self.attention_matrices = attention_matrices

    def get_attn_matrix(self, layer: int, head: int) -> torch.tensor:
        return self.attention_matrices[layer, head]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.inference_mode()
def get_attention_matrices(
    input: str | TokenizerOutput,
    mt: ModelandTokenizer,
    value_weighted: bool = False,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
) -> torch.tensor:
    """
    Parameters:
        prompt: str, input prompt
        mt: ModelandTokenizer, model and tokenizer
        value_weighted: bool.
            - False => will reuturn attention masks for each key-value pair (after softmax). This is the attention mask actually produced inside the model
            - True => will consider the value matrices to give a sense of the actual contribution of source tokens to the target token residual.
    Returns:
        attention matrices: torch.tensor of shape (layers, heads, tokens, tokens)
    """
    # ! doesn't support batching yet. ignore for now
    # TODO (arnab)
    if isinstance(input, str):
        input = prepare_input(prompts=input, tokenizer=mt)
    else:
        assert isinstance(input, TokenizerOutput), (
            "input must be either a string or a TokenizerOutput object"
        )

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])
        return layer_id, int(head_id)

    with mt.trace(input, output_attentions=True):
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) is True:
                    raise NotImplementedError(
                        "patching not supported yet for attn heads"
                    )
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                current_state[:, index, :] = cur_patch.patch
        output = mt.model.output.save()
        logits = mt.output.logits[0][-1].save()

    print(output.keys())
    print(logits.shape)
    attentions = torch.vstack(output.attentions)  # (layers, heads, tokens, tokens)
    if value_weighted:
        values = torch.vstack(
            [output.past_key_values[i][1] for i in range(mt.config.num_hidden_layers)]
        )  # (layers, heads, tokens, head_dim)
        values = repeat_kv(
            values, n_rep=mt.model.layers[0].self_attn.num_key_value_groups
        )
        logger.debug(f"{attentions.shape=} | {values.shape=}")
        attentions = torch.einsum("abcd,abd->abcd", attentions, values.norm(dim=-1))
    return AttentionInformation(
        tokenized_prompt=[mt.tokenizer.decode(tok) for tok in input.input_ids[0]],
        attention_matrices=attentions.detach().cpu().to(torch.float32).numpy(),
        logits=logits.detach().cpu(),
    )

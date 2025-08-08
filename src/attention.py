import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from circuitsvis.tokens import colored_tokens
from dataclasses_json import DataClassJsonMixin
from IPython.display import display

from src.functional import (
    PatchSpec,
    get_module_nnsight,
    patch_with_baukit,
    patch_with_nnsight,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.tokens import find_token_range
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
        assert (
            len(tokenized_prompt) == attention_matrices.shape[-1]
        ), "Tokenized prompt and attention matrices must have the same length"
        assert (
            len(attention_matrices.shape) == 4
        ), "Attention matrices must be of shape (layers, heads, tokens, tokens)"
        assert (
            attention_matrices.shape[-1] == attention_matrices.shape[-2]
        ), "Attention matrices must be square"

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
    patch_interface: callable = patch_with_baukit,
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
        assert isinstance(
            input, TokenizerOutput
        ), "input must be either a string or a TokenizerOutput object"

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        try:
            head_id = int(head_id)
        except ValueError:
            return False
        layer_id = ".".join(module_name.split(".")[:-1])
        return layer_id, int(head_id)

    output = patch_interface(
        mt=mt,
        inputs=input,
        patches=patches if patches is not None else [],
        model_kwargs=dict(output_attentions=True),
    )
    # print(output.__dict__.keys())
    logits = output.logits[0][-1]

    # print(output.keys())
    # print(logits.shape)
    output.attentions = [attn.cuda() for attn in output.attentions]
    attentions = torch.vstack(output.attentions)  # (layers, heads, tokens, tokens)
    if value_weighted:
        values = torch.vstack(
            [
                output.past_key_values[i][1].cuda()
                for i in range(mt.config.num_hidden_layers)
            ]
        )  # (layers, heads, tokens, head_dim)
        values = repeat_kv(
            values, n_rep=mt.model.layers[0].self_attn.num_key_value_groups
        )
        # logger.debug(f"{attentions.shape=} | {values.shape=}")
        attentions = torch.einsum("abcd,abd->abcd", attentions, values.norm(dim=-1))
    return AttentionInformation(
        tokenized_prompt=[mt.tokenizer.decode(tok) for tok in input.input_ids[0]],
        attention_matrices=attentions.detach().cpu().to(torch.float32).numpy(),
        logits=logits.detach().cpu(),
    )


def visualize_attn_matrix(
    attn_matrix: torch.Tensor,
    tokens: list[str],
    q_index: int = -1,
    start_from: int = 1,
):
    assert len(tokens) == attn_matrix.shape[-1]
    attn_matrix = attn_matrix.squeeze()[q_index][start_from:]
    tokens = tokens[start_from:]
    display(colored_tokens(tokens=tokens, values=attn_matrix))


def visualize_average_attn_matrix(
    mt: ModelandTokenizer,
    attn_matrices: dict,
    prompt: str,
    tokenized: Optional[TokenizerOutput] = None,
    layer_window: list | None = None,
    q_index: int = -1,
    remove_bos: bool = True,
    start_from: int | str | None = None,
):
    if tokenized is None:
        tokenized = prepare_input(
            prompts=prompt, tokenizer=mt, return_offsets_mapping=True
        )
    if start_from is None:
        start_from = (
            1
            if remove_bos and tokenized["input_ids"][0][0] == mt.tokenizer.bos_token_id
            else 0
        )
    elif isinstance(start_from, str):
        offset_mapping = (
            tokenized.pop("offset_mapping")[0]
            if "offset_mapping" in tokenized
            else None
        )
        start_from = (
            find_token_range(
                string=prompt.prompt,
                substring="#",
                tokenizer=mt,
                offset_mapping=offset_mapping,
                occurrence=-1,
            )[1]
            - 1
        )

    # print(f"{start_from=}")

    for layer in layer_window:
        print(f"{layer=}")
        if isinstance(attn_matrices, AttentionInformation):
            avg_attn_module_matrix = torch.Tensor(
                attn_matrices.attention_matrices[layer]
            ).mean(dim=0)[q_index]
        else:
            avg_attn_module_matrix = torch.stack(
                [
                    attn_matrices[layer][h_idx].squeeze()
                    for h_idx in range(mt.config.num_attention_heads)
                ]
            ).mean(dim=0)[q_index]

        # print(avg_attn_module_matrix.shape)

        tokens = [
            mt.tokenizer.decode(t, skip_special_tokens=False)
            for t in tokenized["input_ids"][0]
        ][start_from:]
        for idx, t in enumerate(tokens):
            if t == "<think>":
                tokens[idx] = "<|think|>"
            elif t == "</think>":
                tokens[idx] = "<|/think|>"

        display(
            colored_tokens(tokens=tokens, values=avg_attn_module_matrix[start_from:])
        )
        print("-" * 80)

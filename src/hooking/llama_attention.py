import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


@dataclass(frozen=True)
class AttentionEdge:
    #! q_idx *attends* to the k_idx

    q_idx: int
    k_idx: int


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    cut_attn_edges: Optional[dict[int, list[AttentionEdge]]] = None,
    store_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
    freeze_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
    value_weighted: bool = False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.dtype).to(attn_weight.device)

    # print(attn_weight.size())
    # ---------------------------------------------------------------------
    if cut_attn_edges is not None:
        for head_idx, edges in cut_attn_edges.items():
            for edge in edges:
                attn_weight[:, head_idx, edge.q_idx, edge.k_idx] = float("-inf")
    # ---------------------------------------------------------------------

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # ---------------------------------------------------------------------
    if freeze_attn_matrices is not None:
        for head_idx in freeze_attn_matrices:
            assert (
                attn_weight[:, head_idx, :, :].shape
                == freeze_attn_matrices[head_idx].shape
            ), f"Mismatch expected shape {attn_weight[:, head_idx, :, :].shape}, but found shape {freeze_attn_matrices[head_idx].shape}"
            attn_weight[:, head_idx, :, :] = freeze_attn_matrices[head_idx]
    # ---------------------------------------------------------------------

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # print(f"{value_weighted=}")

    # ---------------------------------------------------------------------
    if store_attn_matrices is not None:
        # print(f"{value.shape=}")
        # print(f"{attn_weight.size()=}")
        for head_idx in store_attn_matrices:
            cur_attn_matrix = attn_weight[:, head_idx, :, :]  # batch x q_len x k_len

            if value_weighted is True:
                cur_values = value[:, head_idx, :, :]  # batch x q_len x head_dim

                # * Use this if you want the actual tokenwise contributions as well.
                # cur_attn_matrix = torch.einsum(
                #     "bqk, bqd -> bqkd", cur_attn_matrix, cur_values
                # )

                # * norm of the actual contribution per token (most of the time we want only this)
                cur_attn_matrix = torch.einsum(
                    "bqk, bq -> bqk", cur_attn_matrix, cur_values.norm(dim=-1)
                )
            # print(f"{head_idx} => {cur_attn_matrix.size()=}")

            store_attn_matrices[head_idx] = cur_attn_matrix
    # ---------------------------------------------------------------------

    return attn_weight @ value


def attn_per_head(
    o_proj: torch.nn.modules.linear.Linear,
    attn_output: torch.Tensor,
    freeze_head_contributions: Optional[dict[int, torch.Tensor]] = None,
):
    # print(attn_output.size())
    b, q_len, n_head, h_dim = attn_output.size()
    o_proj_weight_split = o_proj.weight.view(o_proj.out_features, n_head, h_dim)

    print(f"{o_proj_weight_split.size()=}")
    print(f"{attn_output.size()=}")

    per_head_contributions = []
    for head_idx in range(n_head):
        if (
            freeze_head_contributions is not None
            and head_idx in freeze_head_contributions
        ):
            projected_per_head = freeze_head_contributions[head_idx]
        else:
            # calculate the contribution per head
            attn_output_per_head = attn_output[
                :, :, head_idx, :
            ]  # shape: (b, q_len, h_dim)
            attn_output_per_head = attn_output_per_head.to(
                o_proj_weight_split[:, head_idx, :].dtype
            ).to(o_proj_weight_split[:, head_idx, :].device)
            projected_per_head = (
                attn_output_per_head @ o_proj_weight_split[:, head_idx, :].T
            )  # shape: (b, q_len, out_features)

        per_head_contributions.append(projected_per_head)
        # print(f"{projected_per_head.size()=}")

    per_head_contributions = torch.stack(
        per_head_contributions, dim=1
    )  # shape: (b, n_head, q_len, out_features)
    attn_output = per_head_contributions.sum(dim=1)  # shape: (b, q_len, out_features)

    # print(f"{attn_output.size()=} | {per_head_contributions.size()=}")
    return attn_output, per_head_contributions


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    cut_attn_edges = kwargs.get("cut_attn_edges", None)
    store_attn_matrices = kwargs.get("store_attn_matrices", None)
    freeze_attn_matrices = kwargs.get("freeze_attn_matrices", None)
    value_weighted = kwargs.get("value_weighted", False)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    if (
        cut_attn_edges is None
        and store_attn_matrices is None
        and freeze_attn_matrices is None
    ):
        # defer to the default faster implementation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
    else:
        # need to use slower custom implementation, should give numerically similar results
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            cut_attn_edges=cut_attn_edges,
            store_attn_matrices=store_attn_matrices,
            freeze_attn_matrices=freeze_attn_matrices,
            value_weighted=value_weighted,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def LlamaAttentionPatcher(
    block_name: Optional[str] = None,
    cut_attn_edges: Optional[dict[int, list[AttentionEdge]]] = None,
    save_attn_for: Optional[list[int]] = None,
    store_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
    freeze_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
    value_weighted: bool = False,
    store_head_contributions: Optional[dict[int, torch.Tensor]] = None,
    freeze_attn_contributions: Optional[dict[int, torch.Tensor]] = None,
    query_patches: Optional[list[Tuple[int, int, torch.Tensor]]] = None,
) -> callable:
    """
    Wraps the forward method of the `LlamaSdpaAttention` class
    Provides extra arguments for intervention and grabbing attention weights for visualization

    Args:
        block_name: name of the block (mainly for logging and debugging purposes)
        cut_attn_edges: [head_idx, [AttentionEdge(q_idx, k_idx)]] to cut off attention enge q_idx --> k_idx via a specific head
        save_attn_weights: list of head indices to save attention weights for visualization
        attn_matrices: [head_idx, attn_matrix] to store the attention matrix for a specific head
    """

    if save_attn_for is not None:
        assert (
            store_attn_matrices is not None or store_head_contributions is not None
        ), "with save_attn_weights = True you need to provide attn_matrices or attn_contribution"
        if store_attn_matrices is not None:
            assert (
                isinstance(store_attn_matrices, dict) and len(store_attn_matrices) == 0
            )
        if store_head_contributions is not None:
            assert (
                isinstance(store_head_contributions, dict)
                and len(store_head_contributions) == 0
            )

    if store_attn_matrices is not None and store_head_contributions is not None:
        assert save_attn_for is not None

    def forward_patched(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        logger.debug(f"LlamaAttentionPatcher <> {block_name}")

        # if kwargs.get("output_attentions", True):
        #     raise NotImplementedError(
        #         "LlamaAttentionPatcher does not support output_attentions=True."
        #     )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        logger.debug(f"{hidden_shape=} | {input_shape=} | {hidden_states.shape}")

        batch_size, q_len = input_shape
        d_model = hidden_states.size(-1)

        # ---------------------------------------------------------------------
        if save_attn_for is not None:
            # initialize the attn_matrices and attn_contributions with -1
            for head_idx in save_attn_for:
                if store_attn_matrices is not None:
                    store_attn_matrices[head_idx] = (
                        torch.zeros(batch_size, q_len, q_len) - 1
                    )
                if store_head_contributions is not None:
                    store_head_contributions[head_idx] = (
                        torch.zeros(batch_size, q_len, d_model) - 1
                    )
        # ---------------------------------------------------------------------

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        logger.debug(
            f"{query_states.size()=} | {key_states.size()=} | {value_states.size()=}"
        )
        if query_patches is not None:
            for head_idx, token_idx, patch in query_patches:
                query_states[:, head_idx, token_idx, :] = patch

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        assert (
            self.config._attn_implementation == "sdpa"
        ), "NotImplementedError: LlamaAttentionPatcher only supports 'sdpa' implementation"
        attention_interface = sdpa_attention_forward

        kwargs.update(
            {
                "cut_attn_edges": cut_attn_edges,
                "store_attn_matrices": store_attn_matrices,
                "freeze_attn_matrices": freeze_attn_matrices,
                "value_weighted": value_weighted,
            }
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # ---------------------------------------------------------------------
        if (
            store_head_contributions is not None
            or freeze_attn_contributions is not None
        ):
            __attn_output, per_head_contribution = attn_per_head(
                self.o_proj,
                attn_output,
                freeze_head_contributions=freeze_attn_contributions,
            )

        if store_head_contributions is not None:
            for head_idx in store_head_contributions:
                # print(f">>> {head_idx=} | {type(head_idx)}")
                store_head_contributions[head_idx] = per_head_contribution[
                    :, head_idx, :, :
                ]

        if freeze_attn_contributions is not None:
            # logger.warning(
            #     f"{block_name} >> setting modified attn_output to the frozen contributions"
            # )
            attn_output = __attn_output

        else:
            # clean impmementation
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

        # ---------------------------------------------------------------------

        # print(f"{attn_output.size()=}")

        if store_head_contributions is not None:
            if torch.allclose(attn_output, __attn_output, atol=1e-1) is False:
                logger.warning(
                    f"{block_name} >> allclose(attn_output, __attn_output)=False | {attn_output.norm().item()=}, {__attn_output.norm().item()=}"
                )

        return attn_output, attn_weights

    return forward_patched

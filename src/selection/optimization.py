import copy
import json
import logging
import os
import types
from itertools import product
from typing import Any, Literal, Optional

import baukit
import numpy as np
import torch
from torch.optim import AdamW

from src.functional import (
    PatchSpec,
    free_gpu_cache,
    get_hs,
    get_module_nnsight,
    interpret_logits,
    patch_linear_subspaces,
    patch_with_baukit,
)
from src.hooking.llama_attention import LlamaAttentionPatcher
from src.models import ModelandTokenizer
from src.selection.data import (
    CountingSample,
    SelectionSample,
    YesNoSample,
    get_options_for_answer,
)
from src.selection.functional import (
    cache_q_projections,
    find_quesmark_pos,
    get_patches_to_verify_independent_enrichment,
    verify_head_patterns,
    visualize_attn_matrix,
)
from src.selection.utils import get_first_token_id
from src.tokens import prepare_input
from src.utils.typing import PathLike, TokenizerOutput

logger = logging.getLogger(__name__)


# def get_optimal_head_mask(
#     mt: ModelandTokenizer,
#     train_set: list[tuple[SelectionSample, SelectionSample]],
#     learning_rate: float = 1e-3,
#     n_epochs: int = 5,
#     lamb: float = 1e-3,
#     batch_size: int = 4,
#     query_indices: int = [-1],
#     add_ques_pos_to_query_indices: bool = False,
#     black_list_heads: list[
#         tuple[int, int]
#     ] = [],  #! don't consider these heads during training
#     # cache_q_states_before: bool = True,
#     save_path: PathLike | None = None,
#     save_step: int = 5,
# ):
#     hparams = {
#         "learning_rate": learning_rate,
#         "n_epochs": n_epochs,
#         "lamb": lamb,
#         "batch_size": batch_size,
#     }
#     logger.debug(f"Training with hparams: {hparams}")
#     n_layer = mt.n_layer
#     n_heads = mt.config.num_attention_heads

#     mask = torch.ones(
#         (n_layer, n_heads), dtype=mt.dtype, requires_grad=True, device=mt.device
#     )
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # prompts = []
#     # prompts.extend([sample.prompt() for sample in clean_samples])
#     # prompts.extend([sample.prompt() for sample in patch_samples])
#     # tokenized = prepare_input(prompts=prompts, tokenizer=mt)

#     # clean_tokenized = TokenizerOutput(data = {k: v[:len(clean_samples), :] for k, v in tokenized.items()})
#     # patch_tokenized = TokenizerOutput(data = {k: v[len(clean_samples):, :] for k, v in tokenized.items()})

#     optimizer = AdamW([mask], lr=learning_rate)
#     losses = []

#     all_heads = [
#         (layer_idx, head_idx)
#         for layer_idx in range(n_layer)
#         for head_idx in range(n_heads)
#     ]
#     all_q_proj_modules = [
#         mt.attn_module_name_format.format(layer_idx) + ".q_proj"
#         for layer_idx in range(n_layer)
#     ]
#     batches = []
#     for batch_start in range(0, len(train_set), batch_size):
#         batches.append(train_set[batch_start : batch_start + batch_size])

#     def call_cache_projections(
#         clean_samples: list[SelectionSample],
#         patch_samples: list[SelectionSample],
#     ):
#         prompts = []
#         prompts.extend([sample.prompt() for sample in clean_samples])
#         prompts.extend([sample.prompt() for sample in patch_samples])
#         tokenized = prepare_input(
#             prompts=prompts, tokenizer=mt, return_offsets_mapping=True
#         )
#         offset_mapping = tokenized.pop("offset_mapping")
#         patch_tokenized = TokenizerOutput(
#             data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
#         )
#         token_indices = []
#         for idx in range(len(patch_samples)):
#             if add_ques_pos_to_query_indices:
#                 ques_pos = find_quesmark_pos(
#                     prompt=patch_samples[idx].prompt(),
#                     tokenizer=mt.tokenizer,
#                     tokenized=TokenizerOutput(
#                         data={
#                             k: v[idx : idx + 1, :] for k, v in patch_tokenized.items()
#                         }
#                     ),
#                     offset_mapping=offset_mapping[idx + len(clean_samples)],
#                 )
#                 token_indices.append([ques_pos] + copy.deepcopy(query_indices))
#             else:
#                 token_indices.append(copy.deepcopy(query_indices))

#         q_projections = cache_q_projections(
#             mt=mt,
#             input=patch_tokenized,
#             heads=all_heads,
#             token_indices=token_indices,
#             return_output=False,
#         )
#         return q_projections

#     # if cache_q_states_before:
#     #     logger.info("Caching q projections from patch samples...")
#     #     q_projections_from_patch_samples = {}
#     #     for batch_idx, batch in enumerate(batches):
#     #         clean_samples, patch_samples = zip(*batch)

#     #         q_projections = call_cache_projections(
#     #             clean_samples=list(clean_samples),
#     #             patch_samples=list(patch_samples),
#     #         )

#     #         patches = {}
#     #         #! can't do this anymore
#     #         for (layer_idx, head_idx, query_idx), q_proj in q_projections.items():
#     #             module_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
#     #             patches[(module_name, head_idx)] = (layer_idx, q_proj)
#     #         q_projections_from_patch_samples[batch_idx] = patches
#     #         logger.info(f"Caching completed > {batch_idx+1}/{len(batches)} batches.")
#     #         free_gpu_cache()

#     logger.info("Starting training...")

#     head_dim = get_module_nnsight(
#         mt._model, mt.attn_module_name_format.format(0)
#     ).head_dim

#     for epoch in range(n_epochs):
#         epoch_loss = 0
#         for batch_idx, batch in enumerate(batches):
#             optimizer.zero_grad()

#             batch_size_actual = len(batch)

#             clean_samples, patch_samples = zip(*batch)
#             prompts = []
#             prompts.extend([sample.prompt() for sample in clean_samples])
#             prompts.extend([sample.prompt() for sample in patch_samples])
#             tokenized = prepare_input(
#                 prompts=prompts, tokenizer=mt, return_offsets_mapping=True
#             )
#             offset_mapping = tokenized.pop("offset_mapping")
#             clean_tokenized = TokenizerOutput(
#                 data={k: v[: len(clean_samples), :] for k, v in tokenized.items()}
#             )
#             patch_tokenized = TokenizerOutput(
#                 data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
#             )
#             map_int_indices = []
#             for idx in range(len(patch_samples)):
#                 cur_indices = {i: i for i in query_indices}
#                 if add_ques_pos_to_query_indices:
#                     patch_ques_pos = find_quesmark_pos(
#                         prompt=patch_samples[idx].prompt(),
#                         tokenizer=mt.tokenizer,
#                         tokenized=TokenizerOutput(
#                             data={
#                                 k: v[idx : idx + 1, :]
#                                 for k, v in patch_tokenized.items()
#                             }
#                         ),
#                         offset_mapping=offset_mapping[idx + len(clean_samples)],
#                     )
#                     clean_ques_pos = find_quesmark_pos(
#                         prompt=clean_samples[idx].prompt(),
#                         tokenizer=mt.tokenizer,
#                         tokenized=TokenizerOutput(
#                             data={
#                                 k: v[idx : idx + 1, :]
#                                 for k, v in clean_tokenized.items()
#                             }
#                         ),
#                         offset_mapping=offset_mapping[idx],
#                     )
#                     cur_indices[patch_ques_pos] = clean_ques_pos
#                 map_int_indices.append(cur_indices)

#             batch_target_tokens = [
#                 clean_sample.metadata["track_type_obj_token_id"]
#                 for clean_sample in clean_samples
#             ]
#             batch_distractor_tokens = [
#                 [
#                     get_first_token_id(tokenizer=mt.tokenizer, name=option, prefix=" ")
#                     for idx, option in enumerate(clean_sample.options)
#                     if idx != clean_sample.metadata["track_type_obj_idx"]
#                 ]
#                 for clean_sample in clean_samples
#             ]

#             # if cache_q_states_before:
#             #     patch_q_states = q_projections_from_patch_samples[batch_idx]
#             # else:
#             #     q_projections = call_cache_projections(
#             #         clean_samples=list(clean_samples),
#             #         patch_samples=list(patch_samples),
#             #     )
#             #     patches = {}
#             #     for (layer_idx, head_idx, query_idx), q_proj in q_projections.items():
#             #         module_name = (
#             #             mt.attn_module_name_format.format(layer_idx) + ".q_proj"
#             #         )
#             #         patches[(module_name, head_idx)] = (layer_idx, q_proj)
#             #     patch_q_states = patches

#             q_projections = cache_q_projections(
#                 mt=mt,
#                 input=patch_tokenized,
#                 heads=all_heads,
#                 token_indices=[list(mii.keys()) for mii in map_int_indices],
#                 return_output=False,
#             )
#             q_proj_patches = {}
#             for sample_idx in range(batch_size_actual):
#                 sample_patches = {}
#                 for (layer_idx, head_idx, patch_query_idx), q_proj in q_projections[
#                     sample_idx
#                 ].items():
#                     module_name = (
#                         mt.attn_module_name_format.format(layer_idx) + ".q_proj"
#                     )
#                     if (module_name, head_idx) not in sample_patches:
#                         sample_patches[(module_name, head_idx)] = []

#                     sample_patches[(module_name, head_idx)].append(
#                         (
#                             sample_idx,
#                             layer_idx,
#                             head_idx,
#                             map_int_indices[sample_idx][patch_query_idx],
#                             q_proj,
#                         )
#                     )
#                 for lok in sample_patches:
#                     if lok not in q_proj_patches:
#                         q_proj_patches[lok] = []
#                     q_proj_patches[lok].append(sample_patches[lok])

#             # # debug:
#             # head_patch = q_proj_patches[
#             #     (mt.attn_module_name_format.format(35) + ".q_proj", 19)
#             # ]
#             # for sample_idx in range(len(head_patch)):
#             #     for sidx, lidx, hidx, clean_query_idx, q_patch in head_patch[
#             #         sample_idx
#             #     ]:
#             #         print(
#             #             f"{sample_idx} >> {sidx}, {lidx}, {hidx}, {clean_query_idx}, {q_patch.norm():.4f}"
#             #         )

#             batch_size = clean_tokenized.input_ids.shape[0]
#             seq_len = clean_tokenized.input_ids.shape[1]

#             def perform_patch(repr, layer_name):
#                 if layer_name not in all_q_proj_modules:
#                     return repr

#                 repr = repr.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
#                 layer_idx = int(layer_name.split(".")[2])
#                 for head_idx in range(n_heads):
#                     coeff = mask[layer_idx, head_idx].to(repr.dtype).to(repr.device)
#                     for sample_idx in range(batch_size):
#                         sample_patches = q_proj_patches[(layer_name, head_idx)][
#                             sample_idx
#                         ]
#                         for (
#                             sidx,
#                             lidx,
#                             hidx,
#                             clean_query_idx,
#                             q_patch,
#                         ) in sample_patches:
#                             assert sidx == sample_idx
#                             assert lidx == layer_idx
#                             assert hidx == head_idx

#                             q_clean = repr[sample_idx, head_idx, clean_query_idx, :]
#                             q_patch = (
#                                 q_patch.clone().to(q_clean.dtype).to(q_clean.device)
#                             )
#                             q_patch.requires_grad = True
#                             repr[sample_idx, head_idx, clean_query_idx, :] += coeff * (
#                                 q_patch - q_clean
#                             )

#                 repr = repr.transpose(1, 2).view(
#                     batch_size, seq_len, n_heads * head_dim
#                 )
#                 return repr

#             with baukit.TraceDict(
#                 module=mt._model, layers=all_q_proj_modules, edit_output=perform_patch
#             ):
#                 output = mt._model(**clean_tokenized)

#             logits = output.logits[:, -1, :]

#             # calculate target loss
#             target_logits = [
#                 logit[tok] for logit, tok in zip(logits, batch_target_tokens)
#             ]
#             target_loss = -torch.stack(target_logits).mean()  # need this to go up

#             # calculate distractor loss
#             distractor_logits = [
#                 logit[distractor_tokens].mean()
#                 for logit, distractor_tokens in zip(logits, batch_distractor_tokens)
#             ]
#             distractor_loss = torch.stack(distractor_logits).mean()

#             # mask_loss
#             mask_l1_loss = torch.abs(mask).sum() * lamb
#             loss = target_loss + distractor_loss + mask_l1_loss
#             logger.debug(
#                 f"Epoch={epoch+1} | {batch_idx=} |>> {target_loss.item():.4f} + {distractor_loss.item():.4f} + {mask_l1_loss.item():.4f} = {loss.item():.4f}"
#             )

#             loss.backward()
#             optimizer.step()

#             with torch.no_grad():
#                 #! if there are blacklisted heads, set their mask to 0
#                 if black_list_heads:
#                     for layer_idx, head_idx in black_list_heads:
#                         mask[layer_idx, head_idx] = 0.0
#                 mask.clamp_(0, 1)
#                 mask += 1e-3  # to avoid zero gradients

#             epoch_loss += loss.item() * batch_size_actual
#             losses.append(loss.item())

#         epoch_loss /= len(train_set)
#         logger.info(f"Epoch {epoch+1}/{n_epochs} completed. Avg Loss: {epoch_loss:.4f}")
#         mt._model.zero_grad()
#         free_gpu_cache()

#         if save_path is not None and (
#             (epoch + 1) % save_step == 0 or (epoch + 1) == n_epochs
#         ):
#             weight_path = os.path.join(save_path, f"epoch_{epoch+1}.npz")
#             os.makedirs(os.path.dirname(weight_path), exist_ok=True)
#             optimal_mask = mask.round().detach().cpu()
#             np.savez_compressed(
#                 weight_path,
#                 **dict(
#                     optimal_mask=optimal_mask.to(torch.float32).numpy(),
#                     losses=np.array(losses, dtype=np.float32),
#                 ),
#                 allow_pickle=True,
#             )

#     mt._model.zero_grad()
#     with torch.no_grad():
#         mask.clamp_(0, 1)

#     free_gpu_cache()
#     return mask.round().detach().cpu(), losses


def get_optimal_head_mask_optimized(
    mt: ModelandTokenizer,
    train_set: list[tuple[SelectionSample, SelectionSample]],
    learning_rate: float = 1e-3,
    n_epochs: int = 5,
    lamb: float = 1e-3,
    batch_size: int = 4,
    query_indices: int = [-1],
    add_ques_pos_to_query_indices: bool = False,
    black_list_heads: list[tuple[int, int]] = [],
    save_path: PathLike | None = None,
    save_step: int = 5,
    debug_mode: bool = False,  # Added flag for debug prints
):
    """Optimized version with bug fixes and performance improvements"""
    hparams = {
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "lamb": lamb,
        "batch_size": batch_size,
    }
    logger.debug(f"Training with hparams: {hparams}")
    n_layer = mt.n_layer
    n_heads = mt.config.num_attention_heads

    mask = torch.ones(
        (n_layer, n_heads), dtype=mt.dtype, requires_grad=True, device=mt.device
    )

    # Initialize blacklisted heads to 0 from the start
    if black_list_heads:
        with torch.no_grad():
            for layer_idx, head_idx in black_list_heads:
                mask[layer_idx, head_idx] = 0.0

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = AdamW([mask], lr=learning_rate)
    losses = []

    all_heads = [
        (layer_idx, head_idx)
        for layer_idx in range(n_layer)
        for head_idx in range(n_heads)
    ]
    all_q_proj_modules = [
        mt.attn_module_name_format.format(layer_idx) + ".q_proj"
        for layer_idx in range(n_layer)
    ]

    batches = []
    for batch_start in range(0, len(train_set), batch_size):
        batches.append(train_set[batch_start : batch_start + batch_size])

    def call_cache_projections(
        clean_samples: list[SelectionSample],
        patch_samples: list[SelectionSample],
    ):
        prompts = []
        prompts.extend([sample.prompt() for sample in clean_samples])
        prompts.extend([sample.prompt() for sample in patch_samples])
        tokenized = prepare_input(
            prompts=prompts, tokenizer=mt, return_offsets_mapping=True
        )
        offset_mapping = tokenized.pop("offset_mapping")
        patch_tokenized = TokenizerOutput(
            data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
        )
        token_indices = []
        for idx in range(len(patch_samples)):
            if add_ques_pos_to_query_indices:
                ques_pos = find_quesmark_pos(
                    prompt=patch_samples[idx].prompt(),
                    tokenizer=mt.tokenizer,
                    tokenized=TokenizerOutput(
                        data={
                            k: v[idx : idx + 1, :] for k, v in patch_tokenized.items()
                        }
                    ),
                    offset_mapping=offset_mapping[idx + len(clean_samples)],
                )
                token_indices.append([ques_pos] + copy.deepcopy(query_indices))
            else:
                token_indices.append(copy.deepcopy(query_indices))

        q_projections = cache_q_projections(
            mt=mt,
            input=patch_tokenized,
            heads=all_heads,
            token_indices=token_indices,
            return_output=False,
        )
        return q_projections

    logger.info("Starting training...")

    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(batches):
            optimizer.zero_grad()

            # Apply mask adjustments before forward pass
            with torch.no_grad():
                # Ensure mask stays in valid range and avoid zero gradients
                mask.clamp_(0, 1)
                if black_list_heads:
                    for layer_idx, head_idx in black_list_heads:
                        mask[layer_idx, head_idx] = 0.0
                # Add small epsilon to avoid zero gradients (but not to blacklisted heads)
                mask_epsilon = mask.clone()
                mask_epsilon[mask > 0] += 1e-3
                mask.data = mask_epsilon

            batch_size_actual = len(batch)

            clean_samples, patch_samples = zip(*batch)
            prompts = []
            prompts.extend([sample.prompt() for sample in clean_samples])
            prompts.extend([sample.prompt() for sample in patch_samples])
            tokenized = prepare_input(
                prompts=prompts, tokenizer=mt, return_offsets_mapping=True
            )
            offset_mapping = tokenized.pop("offset_mapping")
            clean_tokenized = TokenizerOutput(
                data={k: v[: len(clean_samples), :] for k, v in tokenized.items()}
            )
            patch_tokenized = TokenizerOutput(
                data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
            )

            # Build mapping indices
            map_int_indices = []
            for idx in range(len(patch_samples)):
                cur_indices = {i: i for i in query_indices}
                if add_ques_pos_to_query_indices:
                    patch_ques_pos = find_quesmark_pos(
                        prompt=patch_samples[idx].prompt(),
                        tokenizer=mt.tokenizer,
                        tokenized=TokenizerOutput(
                            data={
                                k: v[idx : idx + 1, :]
                                for k, v in patch_tokenized.items()
                            }
                        ),
                        offset_mapping=offset_mapping[idx + len(clean_samples)],
                    )
                    clean_ques_pos = find_quesmark_pos(
                        prompt=clean_samples[idx].prompt(),
                        tokenizer=mt.tokenizer,
                        tokenized=TokenizerOutput(
                            data={
                                k: v[idx : idx + 1, :]
                                for k, v in clean_tokenized.items()
                            }
                        ),
                        offset_mapping=offset_mapping[idx],
                    )
                    cur_indices[patch_ques_pos] = clean_ques_pos
                map_int_indices.append(cur_indices)

            batch_target_tokens = [
                clean_sample.metadata["track_type_obj_token_id"]
                for clean_sample in clean_samples
            ]
            batch_distractor_tokens = [
                [
                    get_first_token_id(tokenizer=mt.tokenizer, name=option, prefix=" ")
                    for idx, option in enumerate(clean_sample.options)
                    if idx != clean_sample.metadata["track_type_obj_idx"]
                ]
                for clean_sample in clean_samples
            ]

            # Cache projections
            q_projections = cache_q_projections(
                mt=mt,
                input=patch_tokenized,
                heads=all_heads,
                token_indices=[list(mii.keys()) for mii in map_int_indices],
                return_output=False,
            )

            # Pre-process patches for efficient batched application
            q_proj_patches = {}
            for sample_idx in range(batch_size_actual):
                sample_patches = {}
                for (layer_idx, head_idx, patch_query_idx), q_proj in q_projections[
                    sample_idx
                ].items():
                    module_name = (
                        mt.attn_module_name_format.format(layer_idx) + ".q_proj"
                    )
                    if (module_name, head_idx) not in sample_patches:
                        sample_patches[(module_name, head_idx)] = []

                    sample_patches[(module_name, head_idx)].append(
                        (
                            sample_idx,
                            layer_idx,
                            head_idx,
                            map_int_indices[sample_idx][patch_query_idx],
                            q_proj,
                        )
                    )
                for lok in sample_patches:
                    if lok not in q_proj_patches:
                        q_proj_patches[lok] = []
                    q_proj_patches[lok].append(sample_patches[lok])

            # Debug print only if enabled
            if debug_mode and 35 < n_layer and 19 < n_heads:
                key = (mt.attn_module_name_format.format(35) + ".q_proj", 19)
                if key in q_proj_patches:
                    head_patch = q_proj_patches[key]
                    for sample_idx in range(len(head_patch)):
                        for sidx, lidx, hidx, clean_query_idx, q_patch in head_patch[
                            sample_idx
                        ]:
                            print(
                                f"{sample_idx} >> {sidx}, {lidx}, {hidx}, {clean_query_idx}, {q_patch.norm():.4f}"
                            )

            batch_size = clean_tokenized.input_ids.shape[0]
            seq_len = clean_tokenized.input_ids.shape[1]

            def perform_patch_batched(repr, layer_name):
                """Optimized batched version of perform_patch"""
                if layer_name not in all_q_proj_modules:
                    return repr

                # Reshape once
                repr = repr.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                layer_idx = int(layer_name.split(".")[2])

                # Pre-collect all patches for this layer
                updates = []

                for head_idx in range(n_heads):
                    key = (layer_name, head_idx)
                    if key not in q_proj_patches:
                        continue

                    coeff = mask[layer_idx, head_idx].to(repr.dtype).to(repr.device)

                    # Batch process all samples for this head
                    for sample_idx in range(batch_size):
                        if sample_idx >= len(q_proj_patches[key]):
                            continue

                        sample_patches = q_proj_patches[key][sample_idx]

                        for (
                            sidx,
                            lidx,
                            hidx,
                            clean_query_idx,
                            q_patch,
                        ) in sample_patches:
                            assert (
                                sidx == sample_idx
                                and lidx == layer_idx
                                and hidx == head_idx
                            )

                            q_clean = repr[sample_idx, head_idx, clean_query_idx, :]
                            q_patch_tensor = q_patch.to(q_clean.dtype).to(
                                q_clean.device
                            )

                            # Store update info for batched application
                            updates.append(
                                (
                                    sample_idx,
                                    head_idx,
                                    clean_query_idx,
                                    coeff * (q_patch_tensor - q_clean),
                                )
                            )

                # Apply all updates in batch
                if updates:
                    for sample_idx, head_idx, query_idx, delta in updates:
                        repr[sample_idx, head_idx, query_idx, :] += delta

                # Reshape back
                repr = repr.transpose(1, 2).view(
                    batch_size, seq_len, n_heads * head_dim
                )
                return repr

            with baukit.TraceDict(
                module=mt._model,
                layers=all_q_proj_modules,
                edit_output=perform_patch_batched,
            ):
                output = mt._model(**clean_tokenized)

            logits = output.logits[:, -1, :]

            # Calculate target loss
            target_logits = [
                logit[tok] for logit, tok in zip(logits, batch_target_tokens)
            ]
            target_loss = -torch.stack(target_logits).mean()

            # Calculate distractor loss
            distractor_logits = [
                logit[distractor_tokens].mean()
                for logit, distractor_tokens in zip(logits, batch_distractor_tokens)
            ]
            distractor_loss = torch.stack(distractor_logits).mean()

            # Mask L1 regularization
            mask_l1_loss = torch.abs(mask).sum() * lamb
            loss = target_loss + distractor_loss + mask_l1_loss

            logger.debug(
                f"Epoch={epoch+1} | {batch_idx=} |>> {target_loss.item():.4f} + "
                f"{distractor_loss.item():.4f} + {mask_l1_loss.item():.4f} = {loss.item():.4f}"
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Re-apply constraints after optimizer step
                mask.clamp_(0, 1)
                if black_list_heads:
                    for layer_idx, head_idx in black_list_heads:
                        mask[layer_idx, head_idx] = 0.0

            epoch_loss += loss.item() * batch_size_actual
            losses.append(loss.item())

        epoch_loss /= len(train_set)
        logger.info(f"Epoch {epoch+1}/{n_epochs} completed. Avg Loss: {epoch_loss:.4f}")
        mt._model.zero_grad()
        free_gpu_cache()

        if save_path is not None and (
            (epoch + 1) % save_step == 0 or (epoch + 1) == n_epochs
        ):
            weight_path = os.path.join(save_path, f"epoch_{epoch+1}.npz")
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            optimal_mask = mask.round().detach().cpu()
            np.savez_compressed(
                weight_path,
                **dict(
                    optimal_mask=optimal_mask.to(torch.float32).numpy(),
                    losses=np.array(losses, dtype=np.float32),
                ),
                allow_pickle=True,
            )

    mt._model.zero_grad()
    with torch.no_grad():
        mask.clamp_(0, 1)

    free_gpu_cache()
    return mask.round().detach().cpu(), losses


from src.selection.functional import find_quesmark_pos


@torch.inference_mode()
def validate_q_proj_ie_on_sample_pair(
    mt: ModelandTokenizer,
    clean_sample: SelectionSample | CountingSample | YesNoSample,
    patch_sample: SelectionSample | CountingSample | YesNoSample,
    heads: list[tuple[int, int]],
    query_indices: dict[int, int] = {-1: -1},  # patch_idx -> clean_idx
    add_ques_pos_to_query_indices: bool = False,
    verify_head_behavior_on: Optional[int] = None,
    generate_full_ans_for_verify: bool = True,
    ablate_possible_ans_info_from_options: bool = False,
    amplification_scale: float = 1.0,
    must_track_tokens: list[int] = [],
    patch_args: dict[str, Any] = {},
):
    clean_tokenized = prepare_input(
        prompts=clean_sample.prompt(), tokenizer=mt, return_offsets_mapping=True
    )
    patch_tokenized = prepare_input(
        prompts=patch_sample.prompt(), tokenizer=mt, return_offsets_mapping=True
    )
    clean_offset_mapping = clean_tokenized.pop("offset_mapping")[0]
    patch_offset_mapping = patch_tokenized.pop("offset_mapping")[0]
    if add_ques_pos_to_query_indices:
        clean_ques_pos = find_quesmark_pos(
            prompt=clean_sample.prompt(),
            tokenizer=mt.tokenizer,
            tokenized=clean_tokenized,
            offset_mapping=clean_offset_mapping,
        )
        patch_ques_pos = find_quesmark_pos(
            prompt=patch_sample.prompt(),
            tokenizer=mt.tokenizer,
            tokenized=patch_tokenized,
            offset_mapping=patch_offset_mapping,
        )
        query_indices[patch_ques_pos] = clean_ques_pos

    patch_token_indices = [list(query_indices.keys())]
    if patch_args.get("batch_size", 1) > 1:
        patch_samples = []
        logger.debug(f"Sampling {patch_args.get('batch_size', 1)} patch samples...")
        #! Will only work for the SelectOne Task
        # TODO (arnab): fix it
        while len(patch_samples) < patch_args.get("batch_size", 1):
            obj_idx = len(patch_samples) % len(patch_sample.options)
            if patch_args["distinct_options"] is True:
                task = patch_args["task"]
                sample = task.get_random_sample(
                    mt=mt,
                    category=patch_sample.category,
                    prompt_template_idx=patch_args["prompt_template_idx"],
                    option_style=patch_args["option_style"],
                    filter_by_lm_prediction=True,
                    exclude_objs=[clean_sample.obj, patch_sample.obj],
                    n_distractors=patch_args["n_distractors"],
                    obj_idx=obj_idx,
                )
            else:
                sample = copy.deepcopy(patch_sample)
                sample.options[obj_idx], sample.options[sample.obj_idx] = (
                    sample.options[sample.obj_idx],
                    sample.options[obj_idx],
                )
                sample.obj_idx = obj_idx
                # random.shuffle(sample.options)
            patch_samples.append(sample)
        patch_tokenized_batch = prepare_input(
            prompts=[sample.prompt() for sample in patch_samples],
            tokenizer=mt,
            return_offsets_mapping=True,
        )
        patch_offset_mapping_batch = patch_tokenized_batch.pop("offset_mapping")
        patch_token_indices = []
        for idx in range(len(patch_samples)):
            cur_indices = {i: i for i in query_indices}
            if add_ques_pos_to_query_indices:
                patch_ques_pos = find_quesmark_pos(
                    prompt=patch_samples[idx].prompt(),
                    tokenizer=mt.tokenizer,
                    tokenized=TokenizerOutput(
                        data={
                            k: v[idx : idx + 1, :]
                            for k, v in patch_tokenized_batch.items()
                        }
                    ),
                    offset_mapping=patch_offset_mapping_batch[idx],
                )
                cur_indices[patch_ques_pos] = clean_ques_pos
            patch_token_indices.append(list(cur_indices.keys()))
        logger.debug(f"{patch_tokenized_batch.input_ids.shape}")

    if verify_head_behavior_on is not None:
        logger.info("Verifying head behavior...")

        logger.info(
            f"Clean Sample >> Ans: {mt.tokenizer.decode(clean_sample.ans_token_id)}"
        )
        clean_attn_pattern = verify_head_patterns(  # noqa
            prompt=clean_sample.prompt(),
            tokenized_prompt=clean_tokenized,
            # options=clean_sample.options,
            options=[f"{opt}," for opt in clean_sample.options[:-1]]
            + [f"{clean_sample.options[-1]}."],
            mt=mt,
            heads=heads,
            generate_full_answer=generate_full_ans_for_verify,
            query_index=verify_head_behavior_on,
            ablate_possible_ans_info_from_options=ablate_possible_ans_info_from_options,
        )

        logger.info(
            f"Patch Sample >> Ans: {mt.tokenizer.decode(patch_sample.ans_token_id)}"
        )
        patch_attn_pattern = verify_head_patterns(  # noqa
            prompt=patch_sample.prompt(),
            tokenized_prompt=patch_tokenized,
            # options=patch_sample.options,
            options=[f"{opt}," for opt in patch_sample.options[:-1]]
            + [f"{patch_sample.options[-1]}."],
            mt=mt,
            heads=heads,
            generate_full_answer=generate_full_ans_for_verify,
            query_index=verify_head_behavior_on,
            ablate_possible_ans_info_from_options=ablate_possible_ans_info_from_options,
        )

    logger.info(f"Caching the query states for the {len(heads)} heads")

    cached_q_states, patch_output = cache_q_projections(
        mt=mt,
        input=patch_tokenized,
        heads=heads,
        token_indices=[list(query_indices.keys())],
        return_output=True,
    )
    if patch_args.get("batch_size", 1) > 1:
        cached_q_states = cache_q_projections(
            mt=mt,
            input=patch_tokenized_batch,
            heads=heads,
            token_indices=patch_token_indices,
            return_output=False,
        )
        locations = list(cached_q_states[0].keys())
        avg_q_states = {}
        for loc in locations:
            avg_q_states[loc] = torch.stack(
                [cached_q_states[i][loc] for i in range(len(cached_q_states))]
            ).mean(dim=0)
        cached_q_states = [avg_q_states]

    q_proj_patches = []
    for (layer_idx, head_idx, patch_query_idx), q_proj in cached_q_states[0].items():
        q_proj_patches.append(
            PatchSpec(
                location=(
                    mt.attn_module_name_format.format(layer_idx) + ".q_proj",
                    head_idx,
                    query_indices[patch_query_idx],
                ),
                patch=q_proj,
            )
        )

    patch_logits = patch_output.logits[:, -1, :].squeeze()
    patch_predictions = interpret_logits(
        tokenizer=mt,
        logits=patch_logits,
    )
    logger.info(f"patch_prediction={[str(pred) for pred in patch_predictions]}")

    # interested_tokens = [
    #     patch_sample.ans_token_id,
    #     clean_sample.ans_token_id,
    #     clean_sample.metadata["track_type_obj_token_id"],
    # ]
    interested_tokens = get_options_for_answer(clean_sample)
    interested_tokens = [
        get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
        for opt in interested_tokens
    ] + [patch_sample.ans_token_id]
    # interested_tokens += [patch_sample.ans_token_id]
    # interested_tokens = list(set(interested_tokens))  # remove duplicates #! don't need to, made sure during sampling

    logger.info("clean run")
    clean_output = patch_with_baukit(
        mt=mt,
        inputs=clean_tokenized,
        patches=[],
    )
    clean_logits = clean_output.logits[:, -1, :].squeeze()
    clean_predictions, clean_track = interpret_logits(
        tokenizer=mt,
        logits=clean_logits,
        interested_tokens=interested_tokens + must_track_tokens,
    )
    logger.info(f"clean_prediction={[str(pred) for pred in clean_predictions]}")
    logger.info(f"clean_track={clean_track}")

    logger.info("patching the q_proj states")

    if verify_head_behavior_on is not None and amplification_scale == 1.0:
        int_attn_pattern = verify_head_patterns(
            prompt=clean_sample.prompt(),
            tokenized_prompt=clean_tokenized,
            # options=clean_sample.options,
            options=[f"{opt}," for opt in clean_sample.options[:-1]]
            + [f"{clean_sample.options[-1]}."],
            mt=mt,
            heads=heads,
            query_patches=q_proj_patches,
            generate_full_answer=False,
            query_index=verify_head_behavior_on,
            ablate_possible_ans_info_from_options=ablate_possible_ans_info_from_options,
        )
        int_logits = int_attn_pattern["logits"].squeeze()

    else:
        default_attn_implementation = mt.config._attn_implementation
        if amplification_scale != 1.0:
            mt.reset_forward()
            mt.set_attn_implementation("sdpa")

            layers_to_heads = {}
            for layer_idx, head_idx in heads:
                if layer_idx not in layers_to_heads:
                    layers_to_heads[layer_idx] = []
                layers_to_heads[layer_idx].append(head_idx)

            layers_to_q_patches = {}
            for (
                layer_idx,
                head_idx,
                patch_query_idx,
            ), patch in cached_q_states.items():
                if layer_idx not in layers_to_q_patches:
                    layers_to_q_patches[layer_idx] = []
                layers_to_q_patches[layer_idx].append(
                    (head_idx, query_indices[patch_query_idx], patch)
                )

            attention_patterns = {}
            head_contributions = {}
            for layer_idx, head_indices in layers_to_heads.items():
                attn_block_name = mt.attn_module_name_format.format(layer_idx)
                attn_block = baukit.get_module(mt._model, attn_block_name)

                attention_patterns[layer_idx] = {}
                head_contributions[layer_idx] = {}

                attn_block.forward = types.MethodType(
                    LlamaAttentionPatcher(
                        block_name=attn_block_name,
                        save_attn_for=head_indices,
                        store_attn_matrices=attention_patterns[layer_idx],
                        store_head_contributions=head_contributions[layer_idx],
                        query_patches=layers_to_q_patches[layer_idx],
                        amplify_contributions=[
                            (head_idx, q_idx, amplification_scale)
                            for head_idx in head_indices
                            for q_idx in query_indices.values()
                        ],
                        # value_weighted=True,
                    ),
                    attn_block,
                )
            patches = []  # already handled by hooking the default forward pass

        else:
            patches = q_proj_patches

        if ablate_possible_ans_info_from_options:
            patches.extend(
                get_patches_to_verify_independent_enrichment(
                    prompt=clean_sample.prompt(),
                    options=clean_sample.options,
                    pivot=clean_sample.subj,
                    mt=mt,
                    tokenized_prompt=clean_tokenized,
                )
            )

        int_out = patch_with_baukit(
            mt=mt,
            inputs=clean_tokenized,
            patches=patches,
        )
        int_logits = int_out.logits[:, -1, :].squeeze()

        if amplification_scale != 1.0:
            mt.reset_forward()
            mt.set_attn_implementation(default_attn_implementation)

            if verify_head_behavior_on is not None:
                attn_matrix = []
                for layer_idx in attention_patterns:
                    for head_idx in attention_patterns[layer_idx]:
                        attn_matrix.append(
                            attention_patterns[layer_idx][head_idx].cpu()
                        )

                attn_matrix = torch.stack(attn_matrix).squeeze()
                if attn_matrix.dim() == 3:
                    attn_matrix = attn_matrix.mean(dim=0)

                visualize_attn_matrix(
                    attn_matrix=attn_matrix,
                    tokens=[
                        mt.tokenizer.decode(t) for t in clean_tokenized["input_ids"][0]
                    ],
                )

    int_predictions, int_track = interpret_logits(
        tokenizer=mt,
        logits=int_logits,
        interested_tokens=interested_tokens + must_track_tokens,
    )
    logger.info(f"int_prediction={[str(pred) for pred in int_predictions]}")
    logger.info(f"int_track={int_track}")

    return {
        "clean_sample": clean_sample,
        "patch_sample": patch_sample,
        "clean_predictions": clean_predictions,
        "patch_predictions": patch_predictions,
        "int_predictions": int_predictions,
        "clean_track": clean_track,
        "int_track": int_track,
    }


########################### Legacy code below ###########################
# keeping to check performance

from src.functional import repeat_kv


@torch.inference_mode()
def cache_q_projections_prev(
    mt: ModelandTokenizer,
    input: TokenizerOutput,
    query_locations: list[tuple[int, int, int]],  # (layer_idx, head_idx, query_idx)
    return_output: bool = False,
    projection_signature: str = ".q_proj",
):
    layer_to_hq = {}
    for layer_idx, head_idx, query_idx in query_locations:
        if layer_idx not in layer_to_hq:
            layer_to_hq[layer_idx] = []
        layer_to_hq[layer_idx].append((head_idx, query_idx))

    q_projections = {}
    batch_size = input.input_ids.shape[0]
    seq_len = input.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    # head_dim = mt.n_embd // n_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim
    group_size = n_heads // mt.config.num_key_value_heads
    q_module_projections_per_layer = {}
    with mt.trace(input) as tracer:  # noqa
        for layer_idx, query_locs in layer_to_hq.items():
            q_proj_name = (
                mt.attn_module_name_format.format(layer_idx) + projection_signature
            )
            q_proj_module = get_module_nnsight(mt, q_proj_name)
            q_module_projections_per_layer[q_proj_name] = q_proj_module.output.save()

        if return_output:
            output = mt.output.save()

    for layer_idx, query_locs in layer_to_hq.items():
        q_proj_name = (
            mt.attn_module_name_format.format(layer_idx) + projection_signature
        )
        # print(q_proj_name)
        q_proj_out = (
            q_module_projections_per_layer[q_proj_name]
            .view(batch_size, seq_len, -1, head_dim)
            .transpose(1, 2)
        )
        if projection_signature in [".k_proj", ".v_proj"] and group_size != 1:
            q_proj_out = repeat_kv(q_proj_out, n_rep=group_size)
        # print(q_proj_out.shape, q_proj_out.norm())
        for head_idx, query_idx in query_locs:
            q_projections[(layer_idx, head_idx, query_idx)] = (
                q_proj_out[:, head_idx, query_idx, :].clone().squeeze()
            )

    if return_output:
        return q_projections, output
    return q_projections


def promote_target_suppress_distractors(
    mt: ModelandTokenizer,
    source_samples: list,
    destination_samples: list,
    patched_logits: torch.FloatTensor,
):
    batch_target_tokens = [
        destination_sample.metadata["track_type_obj_token_id"]
        for destination_sample in destination_samples
    ]
    batch_distractor_tokens = [
        [
            source_sample.ans_token_id  #! stop from copying the answer from the patch sample
        ]
        + [
            get_first_token_id(tokenizer=mt.tokenizer, name=opt, prefix=" ")
            for opt in get_options_for_answer(destination_sample)
            if get_first_token_id(tokenizer=mt.tokenizer, name=opt, prefix=" ")
            != destination_sample.metadata["track_type_obj_token_id"]
        ]
        for destination_sample, source_sample in zip(
            destination_samples, source_samples
        )
    ]

    target_logits = [
        logit[tok] for logit, tok in zip(patched_logits, batch_target_tokens)
    ]
    target_loss = -torch.stack(target_logits).mean()  # promote target

    # calculate distractor loss
    distractor_logits = [
        logit[distractor_tokens].mean()
        for logit, distractor_tokens in zip(patched_logits, batch_distractor_tokens)
    ]
    distractor_loss = +torch.stack(distractor_logits).mean()  # suppress distractors

    loss = target_loss + distractor_loss

    return loss, {
        "target_loss": target_loss.item(),
        "distractor_loss": distractor_loss.item(),
    }


def match_gold_logit_distribution(
    mt: ModelandTokenizer,
    source_samples: list,
    destination_samples: list,
    patched_logits: torch.FloatTensor,
):
    gold_prompts = []
    for source_sample, destination_sample in zip(source_samples, destination_samples):
        gold_sample = copy.deepcopy(destination_sample)
        gold_sample.category = source_sample.category
        gold_prompts.append(gold_sample.prompt())
    gold_tokenized = prepare_input(prompts=gold_prompts, tokenizer=mt)

    with torch.no_grad():
        gold_output = patch_with_baukit(mt=mt, inputs=gold_tokenized, patches=[])
        gold_logits = gold_output.logits[:, -1, :]

    kldiv_loss = torch.nn.functional.kl_div(
        input=patched_logits.log_softmax(dim=-1),
        target=gold_logits.softmax(dim=-1),
        reduction="batchmean",
    )

    return kldiv_loss, {"kldiv_loss": kldiv_loss.item()}


def increase_logit_in_latents(
    mt: ModelandTokenizer,
    destination_samples: list,
    latents: dict[tuple[str, int], torch.FloatTensor],
    key="track_type_obj",
    **ignored_kwargs,
):
    # logger.debug(f"ignoring kwargs: {ignored_kwargs.keys()}")
    batch_target_tokens = [
        get_first_token_id(
            name=destination_sample.metadata[key], tokenizer=mt.tokenizer, prefix=" "
        )
        for destination_sample in destination_samples
    ]
    # print(batch_target_tokens)

    location_wise_logits = {key: [] for key in latents}
    for (module_name, position), latent in latents.items():
        logit_lens_pred = mt.lm_head(mt.model.norm(latent))  # (batch_size, vocab_size)
        # print(f"{module_name}, {position} >> {logit_lens_pred.shape}")
        batch_target_logits = [
            logit_lens_pred[idx][tok] for idx, tok in enumerate(batch_target_tokens)
        ]
        location_wise_logits[(module_name, position)] = torch.stack(
            batch_target_logits
        ).mean()

    target_loss = -sum(location_wise_logits.values()) / len(
        location_wise_logits
    )  # promote target

    return target_loss, {k: v.item() for k, v in location_wise_logits.items()}


def get_optimal_head_mask_prev(
    mt: ModelandTokenizer,
    train_set: list[tuple[SelectionSample, SelectionSample]],
    learning_rate: float = 1e-3,
    n_epochs: int = 5,
    lamb: float = 1e-3,
    batch_size: int = 4,
    query_indices: int = [-1],
    black_list_heads: list[
        tuple[int, int]
    ] = [],  #! don't consider these heads during training
    cache_q_states_before: bool = False,
    save_path: PathLike | None = None,
    save_step: int = 5,
    loss_fn: Literal[
        "promote_suppress", "match_gold", "increase_logit_in_latents"
    ] = "promote_suppress",
    track_logit_locations: list[tuple[str, int]] | None = None,
):
    hparams = {
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "lamb": lamb,
        "batch_size": batch_size,
        "loss_fn": loss_fn,
    }
    if loss_fn == "increase_logit_in_latents":
        assert track_logit_locations is not None, "track_logit_locations cannot be None"
    loss_fn = {
        "promote_suppress": promote_target_suppress_distractors,
        "match_gold": match_gold_logit_distribution,
        "increase_logit_in_latents": increase_logit_in_latents,
    }[loss_fn]
    logger.debug(f"Training with hparams: {hparams}")
    n_layer = mt.n_layer
    n_heads = mt.config.num_attention_heads

    mask = torch.ones(
        (n_layer, n_heads), dtype=mt.dtype, requires_grad=True, device=mt.device
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # prompts = []
    # prompts.extend([sample.prompt() for sample in clean_samples])
    # prompts.extend([sample.prompt() for sample in patch_samples])
    # tokenized = prepare_input(prompts=prompts, tokenizer=mt)

    # clean_tokenized = TokenizerOutput(data = {k: v[:len(clean_samples), :] for k, v in tokenized.items()})
    # patch_tokenized = TokenizerOutput(data = {k: v[len(clean_samples):, :] for k, v in tokenized.items()})

    optimizer = AdamW([mask], lr=learning_rate)
    losses = []

    all_heads = [
        (layer_idx, head_idx)
        for layer_idx in range(n_layer)
        for head_idx in range(n_heads)
    ]
    all_q_proj_modules = [
        mt.attn_module_name_format.format(layer_idx) + ".q_proj"
        for layer_idx in range(n_layer)
    ]
    batches = []
    for batch_start in range(0, len(train_set), batch_size):
        batches.append(train_set[batch_start : batch_start + batch_size])

    query_locations = [
        (layer_idx, head_idx, query_idx)
        for layer_idx, head_idx in all_heads
        for query_idx in query_indices
    ]

    if cache_q_states_before:
        logger.info("Caching q projections from patch samples...")
        q_projections_from_patch_samples = {}
        for batch_idx, batch in enumerate(batches):
            clean_samples, patch_samples = zip(*batch)
            prompts = []
            prompts.extend([sample.prompt() for sample in clean_samples])
            prompts.extend([sample.prompt() for sample in patch_samples])
            tokenized = prepare_input(
                prompts=prompts,
                tokenizer=mt,
            )
            # clean_tokenized = TokenizerOutput(data = {k: v[:len(clean_samples), :] for k, v in tokenized.items()})
            patch_tokenized = TokenizerOutput(
                data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
            )

            q_projections = cache_q_projections_prev(
                mt=mt,
                input=patch_tokenized,
                query_locations=query_locations,
                return_output=False,
            )

            patches = {}
            for (layer_idx, head_idx, query_idx), q_proj in q_projections.items():
                module_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
                patches[(module_name, head_idx)] = (layer_idx, q_proj)
            q_projections_from_patch_samples[batch_idx] = patches
            logger.info(f"Caching completed > {batch_idx+1}/{len(batches)} batches.")
            free_gpu_cache()

    logger.info("Starting training...")

    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(batches):
            optimizer.zero_grad()

            batch_size_actual = len(batch)

            clean_samples, patch_samples = zip(*batch)
            prompts = []
            prompts.extend([sample.prompt() for sample in clean_samples])
            prompts.extend([sample.prompt() for sample in patch_samples])
            tokenized = prepare_input(prompts=prompts, tokenizer=mt)
            clean_tokenized = TokenizerOutput(
                data={k: v[: len(clean_samples), :] for k, v in tokenized.items()}
            )
            patch_tokenized = TokenizerOutput(
                data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
            )

            if cache_q_states_before:
                patch_q_states = q_projections_from_patch_samples[batch_idx]
            else:
                q_projections = cache_q_projections_prev(
                    mt=mt,
                    input=patch_tokenized,
                    query_locations=query_locations,
                    return_output=False,
                )
                patches = {}
                for (layer_idx, head_idx, query_idx), q_proj in q_projections.items():
                    module_name = (
                        mt.attn_module_name_format.format(layer_idx) + ".q_proj"
                    )
                    patches[(module_name, head_idx)] = (layer_idx, q_proj)
                patch_q_states = patches

            batch_size = clean_tokenized.input_ids.shape[0]
            seq_len = clean_tokenized.input_ids.shape[1]
            # head_dim = mt.n_embd // n_heads

            def perform_patch(repr, layer_name):
                if layer_name not in all_q_proj_modules:
                    return repr
                layer_idx = int(layer_name.split(".")[2])
                repr = repr.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                for head_idx in range(n_heads):
                    coeff = mask[layer_idx, head_idx].to(repr.dtype).to(repr.device)
                    # for query_idx in query_indices:
                    #     q_clean = repr[:, head_idx, query_idx, :]
                    #     layer_idx, q_patch = patch_q_states[(layer_name, head_idx)]
                    #     q_patch = q_patch.clone().to(q_clean.dtype).to(q_clean.device)
                    #     q_patch.requires_grad = True
                    #     # head_patch = coeff * q_patch + (1 - coeff) * q_clean
                    #     repr[:, head_idx, query_idx, :] += coeff * (q_patch - q_clean)
                    q_clean = repr[:, head_idx, query_indices, :]
                    layer_idx, q_patch = patch_q_states[(layer_name, head_idx)]
                    q_patch = q_patch.clone().to(q_clean.dtype).to(q_clean.device)
                    q_patch.requires_grad = True
                    if q_patch.dim() == 2 and q_clean.dim() == 3:
                        q_patch = q_patch.unsqueeze(1)  # Now [batch, 1, head_dim]
                    repr[:, head_idx, query_indices, :] += coeff * (q_patch - q_clean)

                repr = repr.transpose(1, 2).view(
                    batch_size, seq_len, n_heads * head_dim
                )
                return repr

            if track_logit_locations is None:
                with baukit.TraceDict(
                    module=mt._model,
                    layers=all_q_proj_modules,
                    edit_output=perform_patch,
                ):
                    output = mt._model(**clean_tokenized)

                logits = output.logits[:, -1, :]

                target_loss, loss_dict = loss_fn(
                    mt=mt,
                    source_samples=patch_samples,
                    destination_samples=clean_samples,
                    patched_logits=logits,
                )
            else:
                latents = {}
                track_logit_modules = [
                    module_name for module_name, _ in track_logit_locations
                ]
                with baukit.TraceDict(
                    module=mt._model,
                    layers=all_q_proj_modules + track_logit_modules,
                    edit_output=perform_patch,
                ) as trace_dict:
                    output = mt._model(**clean_tokenized)

                latents = {
                    (layer_name, token_idx): trace_dict[layer_name].output[
                        :, token_idx, :
                    ]
                    for layer_name, token_idx in track_logit_locations
                }

                target_loss, loss_dict = loss_fn(
                    mt=mt,
                    destination_samples=clean_samples,
                    latents=latents,
                )

            # mask_loss
            # mask_l1_loss = torch.abs(mask).sum() * lamb
            mask_l1_loss = mask.float().norm(p=1) * lamb  #! testing
            loss = target_loss.float() + mask_l1_loss.to(target_loss.device)
            loss_dict_indv = (
                f"{', '.join([f'{k}={v:.3f}' for k, v in loss_dict.items()])}"
            )
            logger.debug(
                f"Epoch={epoch+1} | {batch_idx=} |>> {target_loss.item():.4f} [{loss_dict_indv}] + {mask_l1_loss.item():.4f} = {loss.item():.4f}"
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                #! if there are blacklisted heads, set their mask to 0
                if black_list_heads:
                    for layer_idx, head_idx in black_list_heads:
                        mask[layer_idx, head_idx] = 0.0
                mask.clamp_(0, 1)
                mask += 1e-4  # to avoid zero gradients

            epoch_loss += loss.item() * batch_size_actual
            losses.append(loss.item())

        epoch_loss /= len(train_set)
        logger.info(f"Epoch {epoch+1}/{n_epochs} completed. Avg Loss: {epoch_loss:.4f}")
        mt._model.zero_grad()
        free_gpu_cache()

        if save_path is not None and (
            (epoch + 1) % save_step == 0 or (epoch + 1) == n_epochs
        ):
            weight_path = os.path.join(save_path, f"epoch_{epoch+1}.npz")
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            optimal_mask = mask.round().detach().cpu()
            np.savez_compressed(
                weight_path,
                **dict(
                    optimal_mask=optimal_mask.to(torch.float32).numpy(),
                    losses=np.array(losses, dtype=np.float32),
                ),
                allow_pickle=True,
            )

    mt._model.zero_grad()
    with torch.no_grad():
        mask.clamp_(0, 1)

    free_gpu_cache()
    return mask.round().detach().cpu(), losses


# for DAS
def get_optimal_rotation(
    mt: ModelandTokenizer,
    train_set: list[tuple[SelectionSample, SelectionSample]],
    layers: list[str],
    token_mapping: dict[int, int] = {-1: -1},  # source_idx -> destination_idx
    rotation_n_dim: int = 128,
    learning_rate: float = 1e-3,
    ortho_reg: float = 0.1,
    n_epochs: int = 5,
    batch_size: int = 4,
    save_path: PathLike | None = None,
    save_step: int = 17,
):
    hparams = {
        "model": mt.name,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "layers": layers,
        "token_mapping": token_mapping,
        "rotation_n_dim": rotation_n_dim,
        "ortho_reg": ortho_reg,
    }
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)
    logger.debug(f"Training with hparams: {hparams}")

    patch_locations = list(product(layers, token_mapping.keys()))
    rotator = {}
    for layer_name in layers:
        module = baukit.get_module(mt._model, layer_name)
        module_device = next(module.parameters()).device
        linear = torch.nn.Linear(mt.n_embd, mt.n_embd, bias=False).to(
            device=module_device
        )
        # initialize as orthogonal
        torch.nn.init.orthogonal_(linear.weight)
        rotator[layer_name] = linear.to(device=module_device, dtype=mt.dtype)

    optimizer = AdamW(
        params=[linear.weight for linear in rotator.values()],
        lr=learning_rate,
        weight_decay=0.0,
    )

    batches = []
    for batch_start in range(0, len(train_set), batch_size):
        batches.append(train_set[batch_start : batch_start + batch_size])

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(batches):
            optimizer.zero_grad()

            batch_size_actual = len(batch)

            destination_samples, source_samples = zip(*batch)
            prompts = []
            prompts.extend([sample.prompt() for sample in destination_samples])
            prompts.extend([sample.prompt() for sample in source_samples])
            tokenized = prepare_input(prompts=prompts, tokenizer=mt)
            destination_tokenized = TokenizerOutput(
                data={k: v[: len(destination_samples), :] for k, v in tokenized.items()}
            )
            source_tokenized = TokenizerOutput(
                data={k: v[len(destination_samples) :, :] for k, v in tokenized.items()}
            )
            # batch_target_tokens = [
            #     clean_sample.metadata["track_type_obj_token_id"]
            #     for clean_sample in destination_samples
            # ]
            # batch_distractor_tokens = [
            #     [
            #         get_first_token_id(tokenizer=mt.tokenizer, name=opt, prefix=" ")
            #         for opt in get_options_for_answer(destination_sample)
            #         if get_first_token_id(tokenizer=mt.tokenizer, name=opt, prefix=" ")
            #         != destination_sample.metadata["track_type_obj_token_id"]
            #     ]
            #     for destination_sample in destination_samples
            # ]

            # print(
            #     f"{source_tokenized.input_ids.shape=}, {destination_tokenized.input_ids.shape=}"
            # )

            gold_prompts = []
            for source_sample, destination_sample in zip(
                source_samples, destination_samples
            ):
                gold_sample = copy.deepcopy(destination_sample)
                gold_sample.category = source_sample.category
                gold_prompts.append(gold_sample.prompt())
            gold_tokenized = prepare_input(prompts=gold_prompts, tokenizer=mt)

            debug = True
            if debug and batch_idx == 0 and epoch == 0:
                print(
                    "source prompt:",
                    source_samples[0].prompt(),
                    ">>",
                    source_samples[0].obj,
                )
                print(
                    "destination prompt:",
                    destination_samples[0].prompt(),
                    ">>",
                    destination_samples[0].obj,
                )
                print("gold prompt:", gold_prompts[0])

            source_hidden_states = get_hs(
                mt=mt,
                input=source_tokenized,
                locations=patch_locations,
                return_dict=True,
            )

            patches = []
            for layer_name, source_idx in patch_locations:
                destination_idx = token_mapping[source_idx]
                source_hs = source_hidden_states[(layer_name, source_idx)]
                patches.append(
                    PatchSpec(
                        location=(layer_name, destination_idx),
                        patch=source_hs,
                    )
                )

            das_patched_output = patch_linear_subspaces(
                mt=mt,
                base_input=destination_tokenized,
                rotator=rotator,
                patches=patches,
                rotate_dimensions=rotation_n_dim,
                with_grad=True,
            )
            logits = das_patched_output.logits[:, -1, :]

            # # calculate target loss
            # target_logits = [
            #     logit[tok] for logit, tok in zip(logits, batch_target_tokens)
            # ]
            # target_loss = -torch.stack(target_logits).mean()  # need this to go up

            # # calculate distractor loss
            # distractor_logits = [
            #     logit[distractor_tokens].mean()
            #     for logit, distractor_tokens in zip(logits, batch_distractor_tokens)
            # ]
            # distractor_loss = torch.stack(distractor_logits).mean()

            # calculate target loss
            with torch.no_grad():
                gold_output = patch_with_baukit(
                    mt=mt, inputs=gold_tokenized, patches=[]
                )
                gold_logits = gold_output.logits[:, -1, :]

            kldiv_loss = torch.nn.functional.kl_div(
                input=logits.log_softmax(dim=-1),
                target=gold_logits.softmax(dim=-1),
                reduction="batchmean",
            )

            # rotator orthogonality loss
            orthogonality_losses = []
            for layer_name in layers:
                identity = torch.eye(mt.n_embd).to(rotator[layer_name].weight.device)
                wt_w = torch.matmul(
                    rotator[layer_name].weight.T, rotator[layer_name].weight
                )
                orthogonality_loss = ((wt_w - identity) ** 2).mean()
                orthogonality_losses.append(orthogonality_loss)
            # ortho_loss = torch.stack(orthogonality_losses).mean().to(target_loss.device)
            ortho_loss = torch.stack(orthogonality_losses).mean().to(kldiv_loss.device)

            # total loss
            # loss = target_loss + distractor_loss + ortho_reg * ortho_loss
            # logger.debug(
            #     f"Epoch={epoch+1} | {batch_idx=} |>> {target_loss.item():.4f} + {distractor_loss.item():.4f} + {ortho_loss.item():.4f} = {loss.item():.4f}"
            # )

            loss = kldiv_loss + ortho_reg * ortho_loss
            logger.debug(
                f"Epoch={epoch+1} | {batch_idx=} |>> {kldiv_loss.item():.4f} + {ortho_loss.item():.4f} = {loss.item():.4f}"
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual
            losses.append(loss.item())

        epoch_loss = epoch_loss / len(train_set)
        logger.info(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.4f}")
        mt._model.zero_grad()
        free_gpu_cache()

        # TODO: save intermediate rotator
        if save_path is not None and (
            (epoch + 1) % save_step == 0 or (epoch + 1) == n_epochs
        ):
            os.makedirs(save_path, exist_ok=True)
            rotator_save_path = os.path.join(save_path, f"epoch_{epoch+1:03d}.pt")
            torch.save(rotator, rotator_save_path)
            logger.info(f"Saved rotator at {rotator_save_path}")

    mt._model.zero_grad()
    for module_name, proj in rotator.items():
        rotator[module_name].requires_grad_(False)

    free_gpu_cache()
    return rotator, losses


@torch.no_grad()
def validate_projections_on_sample_pair(
    mt: ModelandTokenizer,
    destination_sample: SelectionSample | CountingSample | YesNoSample,
    source_sample: SelectionSample | CountingSample | YesNoSample,
    rotate_dimensions: int | Literal["full"],
    rotators: dict[str, torch.nn.Linear | None],
    token_mapping: dict[int, int] = {-1: -1},  # source_idx -> destination_idx
    consider_ques_pos: bool = False,
    must_track_tokens: list[int] = [],
    return_clean_predictions: bool = False,
    debug=False,
):
    if type(rotate_dimensions) is not int:
        assert (
            rotate_dimensions == "full"
        ), "If not int, rotate_dimensions must be 'full'"
    else:
        assert rotate_dimensions > 0, "rotate_dimensions must be positive"

    destination_tokenized = prepare_input(
        prompts=destination_sample.prompt(), tokenizer=mt, return_offsets_mapping=True
    )
    source_tokenized = prepare_input(
        prompts=source_sample.prompt(), tokenizer=mt, return_offsets_mapping=True
    )

    destination_offset_mapping = destination_tokenized.pop("offset_mapping")[0]
    source_offset_mapping = source_tokenized.pop("offset_mapping")[0]
    if consider_ques_pos:
        destination_ques_pos = find_quesmark_pos(
            prompt=destination_sample.prompt(),
            tokenizer=mt.tokenizer,
            tokenized=destination_tokenized,
            offset_mapping=destination_offset_mapping,
        )
        source_ques_pos = find_quesmark_pos(
            prompt=source_sample.prompt(),
            tokenizer=mt.tokenizer,
            tokenized=source_tokenized,
            offset_mapping=source_offset_mapping,
        )
        token_mapping[source_ques_pos] = destination_ques_pos

    ret_dict = {
        "source_sample": source_sample,
        "destination_sample": destination_sample,
    }
    patch_locations = list(product(token_mapping.keys(), rotators.keys()))
    logit_location = (mt.lm_head_name, -1)
    source_hidden_states = get_hs(
        mt=mt,
        input=source_tokenized,
        locations=[(layer_name, token_idx) for token_idx, layer_name in patch_locations]
        + [logit_location],
        return_dict=True,
    )
    if return_clean_predictions or debug:
        source_pred, interested_tokens = interpret_logits(
            tokenizer=mt,
            logits=source_hidden_states[logit_location].squeeze(),
            interested_tokens=[
                get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                for opt in get_options_for_answer(source_sample)
            ]
            + must_track_tokens,
        )
        if return_clean_predictions:
            ret_dict["source_predictions"] = source_pred
            ret_dict["source_track"] = interested_tokens
        if debug:
            logger.debug(
                f"{source_sample.prompt()} >> {mt.tokenizer.decode(source_sample.ans_token_id)}"
            )
            logger.debug(f"Source pred : {[str(pred) for pred in source_pred]}")
            logger.debug(
                f"Source track: {[str(pred) for tok_id, (rank, pred) in interested_tokens.items()]}"
            )

        destination_logit = get_hs(
            mt=mt,
            input=destination_tokenized,
            locations=[logit_location],
            return_dict=False,
        ).squeeze()
        destination_pred, interested_tokens = interpret_logits(
            tokenizer=mt,
            logits=destination_logit,
            interested_tokens=[
                get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                for opt in get_options_for_answer(destination_sample)
            ]
            + must_track_tokens,
        )
        if return_clean_predictions:
            ret_dict["destination_predictions"] = destination_pred
            ret_dict["destination_track"] = interested_tokens
        if debug:
            logger.debug(
                f"{destination_sample.prompt()} >> {mt.tokenizer.decode(destination_sample.ans_token_id)}"
            )
            logger.debug(
                f"Destination pred : {[str(pred) for pred in destination_pred]}"
            )
            logger.debug(
                f"Destination track: {[str(pred) for tok_id, (rank, pred) in interested_tokens.items()]}"
            )

    patches = []
    for source_idx, layer_name in patch_locations:
        destination_idx = token_mapping[source_idx]
        source_hs = source_hidden_states[(layer_name, source_idx)]
        patches.append(
            PatchSpec(
                location=(layer_name, destination_idx),
                patch=source_hs,
            )
        )

    if rotate_dimensions == "full":
        with torch.no_grad():
            patched_output = patch_with_baukit(
                mt=mt,
                inputs=destination_tokenized,
                patches=patches,
            )
    else:
        patched_output = patch_linear_subspaces(
            mt=mt,
            base_input=destination_tokenized,
            rotator=rotators,
            patches=patches,
            rotate_dimensions=rotate_dimensions,
            with_grad=False,
        )

    logits = patched_output.logits[:, -1, :]
    track_tokens = get_options_for_answer(destination_sample)
    track_token_ids = [
        get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
        for opt in track_tokens
    ] + [
        source_sample.ans_token_id
    ]  # also track source ans
    patched_pred, patched_track = interpret_logits(
        tokenizer=mt,
        logits=logits.squeeze(),
        interested_tokens=track_token_ids + must_track_tokens,
    )

    ret_dict["patched_predictions"] = patched_pred
    ret_dict["patched_track"] = patched_track

    if debug:
        logger.debug("-" * 100)
        logger.debug(
            f"target: {destination_sample.metadata['track_type_obj']} | \"{mt.tokenizer.decode(destination_sample.metadata['track_type_obj_token_id'])}\""
        )
        logger.debug(f"Patched pred : {[str(pred) for pred in patched_pred]}")
        logger.debug(
            f"Patched track: {[str(pred) for tok_id, (rank, pred) in patched_track.items()]}"
        )
        logger.debug("-" * 100)

    return ret_dict


# for DCM on the SVD (q projection) components
def apply_q_proj_patch_with_projection(
    mt: ModelandTokenizer,
    source_tokenized: TokenizerOutput,
    destination_tokenized: TokenizerOutput,
    projections: dict[tuple[int, int], torch.Tensor],
    token_indices: list[int],
):
    q_proj_modules = []
    layer_to_heads = {}
    query_locations = []
    for layer_idx, head_idx in projections.keys():
        module_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
        q_proj_modules.append(module_name)
        if layer_idx not in layer_to_heads:
            layer_to_heads[layer_idx] = []
        layer_to_heads[layer_idx].append(head_idx)
        query_locations.extend(
            (layer_idx, head_idx, query_idx) for query_idx in token_indices
        )

    q_projections = cache_q_projections_prev(
        mt=mt,
        input=source_tokenized,
        query_locations=query_locations,
        return_output=False,
    )
    patches = {}
    for (layer_idx, head_idx, query_idx), q_proj in q_projections.items():
        module_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
        patches[(module_name, head_idx)] = (layer_idx, q_proj)

    patch_q_states = patches
    batch_size = destination_tokenized.input_ids.shape[0]
    seq_len = destination_tokenized.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim

    def perform_patch(repr, layer_name):
        if layer_name not in q_proj_modules:
            return repr
        # logger.debug(f"Patching at layer: {layer_name}")
        layer_idx = int(layer_name.split(".")[2])
        repr = repr.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        for head_idx in layer_to_heads[layer_idx]:
            if (layer_idx, head_idx) not in projections:
                assert (
                    False
                ), f"{(layer_idx, head_idx)} not in projections. This should never happen!"
            projection = projections[(layer_idx, head_idx)]
            q_clean = repr[:, head_idx, token_indices, :]
            layer_idx, q_patch = patch_q_states[(layer_name, head_idx)]
            q_patch = q_patch.clone().to(q_clean.dtype).to(q_clean.device)
            if q_patch.dim() == 2 and q_clean.dim() == 3:
                q_patch = q_patch.unsqueeze(1)  # Now [batch, 1, head_dim]
            q_patch_proj = q_patch @ projection
            q_clean_proj = q_clean @ projection
            repr[:, head_idx, token_indices, :] += q_patch_proj - q_clean_proj

        repr = repr.transpose(1, 2).view(batch_size, seq_len, n_heads * head_dim)
        return repr

    with baukit.TraceDict(
        module=mt._model, layers=q_proj_modules, edit_output=perform_patch
    ):
        output = mt._model(**destination_tokenized)

    return output


def get_optimal_component_mask(
    mt: ModelandTokenizer,
    train_set: list[tuple[SelectionSample, SelectionSample]],
    q_proj_basis_directions: dict[tuple[int, int], torch.Tensor],
    learning_rate: float = 1e-3,
    n_epochs: int = 5,
    lamb: float = 1e-3,
    batch_size: int = 4,
    query_indices: int = [-1],
    save_path: PathLike | None = None,
    save_step: int = 5,
    loss_fn: Literal["promote_suppress", "match_gold"] = "match_gold",
):
    hparams = {
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "lamb": lamb,
        "batch_size": batch_size,
        "loss_fn": loss_fn,
    }
    loss_fn = {
        "promote_suppress": promote_target_suppress_distractors,
        "match_gold": match_gold_logit_distribution,
    }[loss_fn]
    logger.debug(f"Training with hparams: {hparams}")
    # n_layer = mt.n_layer
    # n_heads = mt.config.num_attention_heads
    head_dim = get_module_nnsight(
        mt._model, mt.attn_module_name_format.format(0)
    ).head_dim

    masks = {}
    for layer_idx, head_idx in q_proj_basis_directions.keys():
        masks[(layer_idx, head_idx)] = torch.ones(
            (head_dim,), dtype=mt.dtype, requires_grad=True, device=mt.device
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = AdamW([mask for mask in masks.values()], lr=learning_rate)
    losses = []

    all_q_proj_modules = []
    query_locations = []
    all_heads = list(q_proj_basis_directions.keys())
    for layer_idx, head_idx in all_heads:
        module_name = mt.attn_module_name_format.format(layer_idx) + ".q_proj"
        all_q_proj_modules.append(module_name)
        query_locations.extend(
            (layer_idx, head_idx, query_idx) for query_idx in query_indices
        )

    batches = []
    for batch_start in range(0, len(train_set), batch_size):
        batches.append(train_set[batch_start : batch_start + batch_size])

    def build_projections(masks):
        projections = {}
        for layer_idx, head_idx in all_heads:
            basis_directions = q_proj_basis_directions[(layer_idx, head_idx)]
            mask = (
                masks[(layer_idx, head_idx)]
                .to(basis_directions.dtype)
                .to(basis_directions.device)
            )
            masked_basis = basis_directions * mask[:, None]
            projections[(layer_idx, head_idx)] = masked_basis.T @ masked_basis
        return projections

    @torch.no_grad()
    def save_projections(save_file: PathLike):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        optimal_masks = {key: mask.clone().round() for key, mask in masks.items()}
        with torch.no_grad():
            final_projections = build_projections(optimal_masks)
        torch.save(
            {
                "projections": final_projections,
                "masks": optimal_masks,
                "hparams": hparams,
            },
            save_file,
        )
        del optimal_masks, final_projections
        free_gpu_cache()
        return

    logger.info("Starting training...")

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(batches):
            optimizer.zero_grad()

            batch_size_actual = len(batch)

            clean_samples, patch_samples = zip(*batch)
            prompts = []
            prompts.extend([sample.prompt() for sample in clean_samples])
            prompts.extend([sample.prompt() for sample in patch_samples])
            tokenized = prepare_input(prompts=prompts, tokenizer=mt)
            clean_tokenized = TokenizerOutput(
                data={k: v[: len(clean_samples), :] for k, v in tokenized.items()}
            )
            patch_tokenized = TokenizerOutput(
                data={k: v[len(clean_samples) :, :] for k, v in tokenized.items()}
            )

            projections = build_projections(masks=masks)

            output = apply_q_proj_patch_with_projection(
                mt=mt,
                source_tokenized=patch_tokenized,
                destination_tokenized=clean_tokenized,
                projections=projections,
                token_indices=query_indices,
            )

            logits = output.logits[:, -1, :]

            target_loss, loss_dict = loss_fn(
                mt=mt,
                source_samples=patch_samples,
                destination_samples=clean_samples,
                patched_logits=logits,
            )

            # mask loss
            mask_l1_loss = None
            for mask in masks.values():
                mask = mask.float()
                if mask_l1_loss is None:
                    mask_l1_loss = lamb * mask.norm(p=1)
                else:
                    mask_l1_loss += lamb * mask.norm(p=1).to(mask_l1_loss.device)

            loss = target_loss.float() + mask_l1_loss
            # loss = mask_l1_loss
            loss_dict_indv = (
                f"{', '.join([f'{k}={v:.3f}' for k, v in loss_dict.items()])}"
            )
            logger.debug(
                f"Epoch={epoch+1} | {batch_idx=} |>> {target_loss.item():.4f} [{loss_dict_indv}] + {mask_l1_loss.item():.4f} = {loss.item():.4f}"
            )

            loss.backward()
            # checking if gradients are flowing
            # for key, mask in list(masks.items())[:5]:
            #     if mask.grad is not None:
            #         print(f"{key}: grad norm = {mask.grad.norm().item():.6f}")
            #     else:
            #         print(f"{key}: NO GRADIENT!")
            optimizer.step()

            with torch.no_grad():
                for mask in masks.values():
                    mask.clamp_(0, 1)
                    # mask += 1e-4  # to avoid zero gradients

            # print(f"Mask sample values: {list(masks.values())[0][:5]}")  # First 5 elements
            # print(f"Mask mean: {list(masks.values())[0].mean().item()}")

            epoch_loss += loss.item() * batch_size_actual
            losses.append(loss.item())

        epoch_loss /= len(train_set)
        logger.info(f"Epoch {epoch+1}/{n_epochs} completed. Avg Loss: {epoch_loss:.4f}")

        mt._model.zero_grad()
        del (
            projections,
            output,
            logits,
        )
        free_gpu_cache()

        if save_path is not None and (
            (epoch + 1) % save_step == 0 or (epoch + 1) == n_epochs
        ):
            weight_path = os.path.join(save_path, f"epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            save_projections(save_file=weight_path)
            logger.info(f"Saved optimal projections to {weight_path}")

    final_masks = {key: mask.detach().round() for key, mask in masks.items()}
    final_projections = build_projections(final_masks)
    free_gpu_cache()
    return final_projections, final_masks, losses


@torch.no_grad()
def validate_low_rank_svd_bases_on_sample_pair(
    mt: ModelandTokenizer,
    destination_sample: SelectionSample,
    source_sample: SelectionSample,
    projections: dict[tuple[int, int], torch.Tensor],
    token_indices: list[int] = [-1],  # source_idx -> destination_idx
    must_track_tokens: list[int] = [],
    return_clean_predictions: bool = False,
    debug=False,
):
    destination_tokenized = prepare_input(
        prompts=destination_sample.prompt(),
        tokenizer=mt,
        # return_offsets_mapping=True
    )
    source_tokenized = prepare_input(
        prompts=source_sample.prompt(),
        tokenizer=mt,
        # return_offsets_mapping=True
    )

    # destination_offset_mapping = destination_tokenized.pop("offset_mapping")[0]
    # source_offset_mapping = source_tokenized.pop("offset_mapping")[0]

    ret_dict = {
        "source_sample": source_sample,
        "destination_sample": destination_sample,
    }
    logit_location = (mt.lm_head_name, -1)

    if return_clean_predictions or debug:
        source_hidden_states = get_hs(
            mt=mt,
            input=source_tokenized,
            locations=[logit_location],
            return_dict=True,
        )
        source_pred, interested_tokens = interpret_logits(
            tokenizer=mt,
            logits=source_hidden_states[logit_location].squeeze(),
            interested_tokens=[
                get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                for opt in get_options_for_answer(source_sample)
            ]
            + must_track_tokens,
        )
        if return_clean_predictions:
            ret_dict["source_predictions"] = source_pred
            ret_dict["source_track"] = interested_tokens
        if debug:
            logger.debug(
                f"{source_sample.prompt()} >> {mt.tokenizer.decode(source_sample.ans_token_id)}"
            )
            logger.debug(f"Source pred : {[str(pred) for pred in source_pred]}")
            logger.debug(
                f"Source track: {[str(pred) for tok_id, (rank, pred) in interested_tokens.items()]}"
            )

        destination_logit = get_hs(
            mt=mt,
            input=destination_tokenized,
            locations=[logit_location],
            return_dict=False,
        ).squeeze()
        destination_pred, interested_tokens = interpret_logits(
            tokenizer=mt,
            logits=destination_logit,
            interested_tokens=[
                get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                for opt in get_options_for_answer(destination_sample)
            ]
            + must_track_tokens,
        )
        if return_clean_predictions:
            ret_dict["destination_predictions"] = destination_pred
            ret_dict["destination_track"] = interested_tokens
        if debug:
            logger.debug(
                f"{destination_sample.prompt()} >> {mt.tokenizer.decode(destination_sample.ans_token_id)}"
            )
            logger.debug(
                f"Destination pred : {[str(pred) for pred in destination_pred]}"
            )
            logger.debug(
                f"Destination track: {[str(pred) for tok_id, (rank, pred) in interested_tokens.items()]}"
            )

    patched_output = apply_q_proj_patch_with_projection(
        mt=mt,
        source_tokenized=source_tokenized,
        destination_tokenized=destination_tokenized,
        projections=projections,
        token_indices=token_indices,
    )

    logits = patched_output.logits[:, -1, :]
    track_tokens = get_options_for_answer(destination_sample)
    track_token_ids = [
        get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
        for opt in track_tokens
    ] + [
        source_sample.ans_token_id
    ]  # also track source ans
    patched_pred, patched_track = interpret_logits(
        tokenizer=mt,
        logits=logits.squeeze(),
        interested_tokens=track_token_ids + must_track_tokens,
    )

    ret_dict["patched_predictions"] = patched_pred
    ret_dict["patched_track"] = patched_track

    if debug:
        logger.debug("-" * 100)
        logger.debug(
            f"target: {destination_sample.metadata['track_type_obj']} | \"{mt.tokenizer.decode(destination_sample.metadata['track_type_obj_token_id'])}\""
        )
        logger.debug(f"Patched pred : {[str(pred) for pred in patched_pred]}")
        logger.debug(
            f"Patched track: {[str(pred) for tok_id, (rank, pred) in patched_track.items()]}"
        )
        logger.debug("-" * 100)

    return ret_dict

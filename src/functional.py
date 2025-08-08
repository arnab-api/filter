import copy
import gc
import logging
import re
import string
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import baukit
import numpy as np
import torch
from nltk.corpus import stopwords
from nnsight import LanguageModel

from src.dataset import Relation
from src.models import ModelandTokenizer
from src.tokens import find_token_range, insert_padding_before_pos, prepare_input
from src.utils.typing import (
    SVD,
    ArrayLike,
    Model,
    PredictedToken,
    Tokenizer,
    TokenizerOutput,
)

logger = logging.getLogger(__name__)


def get_keywords_from_text(
    text: str,
    tokenizer: Tokenizer | ModelandTokenizer,
    maybe_prepend_space: bool = True,
) -> list[int]:
    tokenizer = unwrap_tokenizer(tokenizer)
    if maybe_prepend_space is True and text.startswith(" ") is False:
        text = f" {text}"
    tokenized = tokenizer(text, add_special_tokens=False).input_ids
    # print([tokenizer.decode(t) for t in tokenized])
    filtered = []
    prev_tok = " "
    for idx, t_idx in enumerate(tokenized):
        tok = tokenizer.decode(t_idx)
        skip = False
        if t_idx in tokenizer.all_special_ids:
            skip = True
        if tok in string.whitespace:
            skip = True
        if tok.strip() in string.punctuation:
            skip = True
        if tok.strip().lower() in stopwords.words("english"):
            # print(tokenizer.decode(tokenized[idx + 1]))
            if idx < len(tokenized) - 1 and tokenizer.decode(
                tokenized[idx + 1]
            ).startswith(" "):
                skip = True
        if (
            prev_tok.endswith(" ") is False and tok.startswith(" ") is False
        ):  # continuation of a word, safe to ignore?
            skip = True

        if skip is False:
            filtered.append(t_idx)

        prev_tok = tok
    return filtered


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
    interested_tokens: tuple[int] = (),
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    tokenizer = unwrap_tokenizer(tokenizer)
    # print(type(tokenizer))
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    if isinstance(top_k_indices, ArrayLike) is False:
        top_k_indices = [top_k_indices]

    candidates = [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]

    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        # print(interested_logits)
        interested_logits = {
            k: v
            for k, v in sorted(
                interested_logits.items(), key=lambda x: x[1][1].prob, reverse=True
            )
        }
        return candidates, interested_logits
    return candidates


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: tuple[int] = (),
    k: int = 5,
    return_logits=False,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    inputs = mt.tokenizer(
        mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    )
    with mt.trace(inputs):
        lnf = get_module_nnsight(mt, mt.final_layer_norm_name)
        lnf.input = h.view(1, 1, h.squeeze().shape[0])
        logits = mt.output.logits.save()

    logits = logits.squeeze()
    free_gpu_cache()
    ret = interpret_logits(
        tokenizer=mt, logits=logits, k=k, interested_tokens=interested_tokens
    )

    ret = (logits, ret) if return_logits else ret

    return ret


@torch.inference_mode()
def logit_lens_baukit(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: tuple[int] = (),
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    lnf = baukit.get_module(mt._model, mt.final_layer_norm_name)
    lm_head = baukit.get_module(mt._model, mt.lm_head_name)
    h = untuple(h)
    logits = lm_head(lnf(h)).squeeze()
    free_gpu_cache()
    return interpret_logits(
        tokenizer=mt, logits=logits, k=k, interested_tokens=interested_tokens
    )


@torch.inference_mode()
def forward_pass_to_vocab(
    mt: ModelandTokenizer, h: torch.Tensor, layer_name: str, **kwargs
):
    inputs = mt.tokenizer(
        mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    )
    with mt.trace(inputs):
        module = get_module_nnsight(mt, layer_name)
        module.output[0][0, :] = h
        logits = mt.output.logits[0, -1].save()

    free_gpu_cache()

    return interpret_logits(
        tokenizer=mt,
        logits=logits,
        **kwargs,
    )


@torch.inference_mode()
def patchscope(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    context: str | None = None,
    placeholder: str = "x",
    context_tokenized: TokenizerOutput | None = None,
    placeholder_idx: int | None = None,
    patch_layers: Optional[list[str]] = None,
    return_logits: bool = False,
    add_orig_latent_to: str | None = None,
    **interpret_kwargs,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    if context is None and context_tokenized is None:
        phrases = [
            " copy",
            " Cat",
            " Java",
            " transistor",
            " python",
            " Leonardo DiCaprio",
            " The Lion King",
            " Washington D.C.",
            " Mount Everest",
            " computer",
        ]
        context = "\n".join([f"{p} >{p}" for p in phrases])
        context = f"{context}\n {placeholder} >"
        logger.debug(context)

    if context_tokenized is None:
        context_tokenized = prepare_input(
            tokenizer=mt,
            prompts=context,
            return_offsets_mapping=True,
        )

    elif context is None:
        context = mt.tokenizer.decode(
            context_tokenized["input_ids"][0], skip_special_tokens=True
        )
        logger.debug(f"context: {context}")

    if placeholder_idx is None:
        placeholder_range = find_token_range(
            string=context,
            substring=placeholder,
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=(
                context_tokenized["offset_mapping"][0]
                if "offset_mapping" in context_tokenized
                else None
            ),
        )
        placeholder_idx = placeholder_range[1] - 1
        # print(context_tokenized)
        # logger.debug(
        #     f"placeholder position: {placeholder_idx} | token: \"{mt.tokenizer.decode(context_tokenized['input_ids'][0, placeholder_idx])}\""
        # )

    if "offset_mapping" in context_tokenized:
        context_tokenized.pop("offset_mapping")

    placeholder_ends = mt.tokenizer.decode(
        context_tokenized["input_ids"][0, placeholder_idx]
    )
    if placeholder.strip().endswith(placeholder_ends) is False:
        logger.warning(
            f"{placeholder=} does not end with {placeholder_ends=} | {placeholder_idx=}"
        )

    patch_layers = (
        [mt.layer_name_format.format(5)] if patch_layers is None else patch_layers
    )
    patches = [
        PatchSpec(
            location=(layer_name, placeholder_idx),
            patch=h,
            clean=None,
            strategy="replace",
        )
        for layer_name in patch_layers
    ]
    if add_orig_latent_to is not None:
        # print("adding")
        patches.append(
            PatchSpec(
                location=(add_orig_latent_to, placeholder_idx),
                patch=h,
                clean=None,
                strategy="add",
            )
        )

    # print(patches)

    logits = get_hs(
        mt=mt,
        input=context_tokenized,
        locations=[(mt.lm_head_name, -1)],
        patches=patches,
        return_dict=False,
    )
    pred = interpret_logits(
        tokenizer=mt,
        logits=logits,
        **interpret_kwargs,
    )

    if return_logits:
        return logits, pred
    return pred


def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | Tokenizer) -> Tokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


# useful for logging
def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def filter_samples_by_model_knowledge(
    mt: ModelandTokenizer, relation: Relation, limit: Optional[int] = None
) -> Relation:
    """Filter samples by model knowledge."""
    logger.debug(f'"{relation.name}" | filtering with {mt.name}')

    filtered_samples = []
    for i in range(len(relation.samples)):
        question, answer = relation[i]
        predictions = predict_next_token(mt, question, k=5)[0]
        top_pred = predictions[0]
        is_known = is_nontrivial_prefix(prediction=top_pred.token, target=answer)
        sample = relation.samples[i]

        logger.debug(
            f"{sample.subject=} -> {answer=} | predicted = '{top_pred.token}'({top_pred.prob:.3f}) ==> ({get_tick_marker(is_known)})"
        )
        if is_known:
            filtered_samples.append(sample)

        if limit is not None and len(filtered_samples) >= limit:
            break

    logger.info(
        f'filtered relation "{relation.name}" to {len(filtered_samples)} samples (with {len(relation._few_shot_samples)}-shots)'
    )

    relation.samples = filtered_samples
    return relation


def release_cache():
    gc.collect()
    if torch.cuda.is_available():
        # before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        # after = torch.cuda.memory_allocated()
        # freed = before - after # ! the effect of empty_cache() is not immediate


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None
    strategy: Literal["replace", "add"] = "replace"


def generate_with_patch(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    n_gen_per_prompt: int = 5,
    max_new_tokens: int = 20,
    patches: Optional[list[PatchSpec]] = None,
    do_sample: bool = True,
    patch_strategy: Literal["replace", "add"] = "replace",
    patch_at_all_generations: bool = False,
    remove_prefix: bool = False,
    **kwargs,
) -> list[str]:
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(
            prompts=[inputs],
            tokenizer=mt,
            n_gen_per_prompt=n_gen_per_prompt,
        )

    with mt.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
        **kwargs,
    ) as gen_trace:  # noqa: F841
        if patches is not None:
            if patch_at_all_generations:
                mt.all()
            for cur_patch in patches:
                module_name, index = cur_patch.location
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                if current_state.ndim == 2:
                    current_state = current_state.unsqueeze(0)
                if patch_strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif patch_strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError("patch_strategy must be one of 'replace', 'add'")
        gen_out = mt.generator.output.save()

    start = 0
    if remove_prefix:
        start = inputs.input_ids.shape[1]
    return mt.tokenizer.batch_decode(
        gen_out.sequences[:, start:], skip_special_tokens=True
    )


@torch.inference_mode()
def generate_with_beam_search(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    num_beams: int = 20,
    num_return_sequences: int = 10,
    max_new_tokens: int = 20,
    no_repeat_ngram_size: int = 3,
    processor: callable = lambda x: x,
) -> list[str]:
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(
            prompts=[inputs],
            tokenizer=mt,
        )

    if no_repeat_ngram_size < 1:
        logger.warning("no_repeat_ngram_size >= 1 recommended for varied generations")

    outputs = mt._model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        output_scores=True,
        return_dict_in_generate=True,
    )
    start = inputs.input_ids.shape[1]
    generations = mt.tokenizer.batch_decode(
        outputs.sequences[:, start:], skip_special_tokens=True
    )
    return [processor(g) for g in generations]


@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    inputs: Union[str, list[str]] | TokenizerOutput,
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
):
    """Predict the next token(s) given the input."""
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = [inputs] if isinstance(inputs, str) else inputs
    if token_of_interest is not None:
        token_of_interest = (
            [token_of_interest]
            if not isinstance(token_of_interest, list)
            else token_of_interest
        )
    if token_of_interest is not None:
        # print(f"{len(token_of_interest)=} | {len(inputs)=}")
        if isinstance(token_of_interest, ArrayLike) is False:
            token_of_interest = [token_of_interest]
        if isinstance(token_of_interest[0], ArrayLike) is False:
            token_of_interest = [token_of_interest]

        assert len(token_of_interest) == (
            len(inputs["input_ids"])
            if isinstance(inputs, TokenizerOutput)
            else len(inputs)
        )
        track_interesting_tokens = []

    is_tokenized = isinstance(inputs, TokenizerOutput)
    total_len = len(inputs["input_ids"]) if is_tokenized else len(inputs)

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]
        if total_len > 1:
            logger.warning(
                "passed `patches`, not supported for batched predictions yet. will give weird results."
            )

    predictions = []
    for i in range(0, total_len, batch_size):
        if is_tokenized is False:
            batch_inputs = prepare_input(
                tokenizer=mt,
                prompts=inputs[i : i + batch_size],
                padding_side="left",
            )
        else:
            batch_inputs = {
                k: v[i : i + batch_size] if isinstance(v, ArrayLike) else v
                for k, v in inputs.items()
            }

        # print(i, i + batch_size, batch_inputs["input_ids"].shape)

        def is_an_attn_head(module_name) -> bool | tuple[int, int]:
            attn_id = mt.attn_module_name_format.split(".")[-1]
            if attn_id not in module_name:
                return False
            if module_name.endswith(attn_id):
                return False

            head_id = module_name.split(".")[-1]
            layer_id = ".".join(module_name.split(".")[:-1])
            return layer_id, int(head_id)

        with mt.trace(batch_inputs, scan=False, validate=False) as tr:  # noqa: F841
            # TODO: patching code is being repeated a couple of times. refactor it.
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
            batch_logits = mt.output.logits.save()

        batch_logits = batch_logits[:, -1, :]
        batch_probs = batch_logits.float().softmax(dim=-1)
        batch_topk = batch_probs.topk(k=k, dim=-1)

        batch_interested_token_indices = None
        if token_of_interest is not None:
            batch_interested_token_indices = []
            for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                batch_interested_token_indices.append(
                    mt.tokenizer(
                        token_of_interest[j],
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids[:, 0]
                    if type(token_of_interest[j]) is str
                    else token_of_interest[j]
                )

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            top_pred = interpret_logits(
                tokenizer=mt,
                logits=batch_logits[batch_order],
                k=k,
                interested_tokens=(
                    batch_interested_token_indices[batch_order]
                    if token_of_interest is not None
                    else []
                ),
            )
            if token_of_interest is not None:
                track_interesting_tokens.append(top_pred[1])
                top_pred = top_pred[0]

            predictions.append(top_pred)

        free_gpu_cache()

    if token_of_interest is not None:
        return predictions, track_interesting_tokens

    return predictions


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


def patch_with_baukit(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list[PatchSpec] = [],
    model_kwargs: dict = {},
):
    model = mt._model

    batch_size = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    head_dim = mt.config.hidden_size // n_heads

    modules_to_patches = {}
    for patch in patches:
        if len(patch.location) == 2:
            layer, idx = patch.location
            head_idx = None
        elif len(patch.location) == 3:
            layer, head_idx, idx = patch.location
        else:
            raise ValueError(f"Invalid patch location: {patch.location}")
        if layer not in modules_to_patches:
            modules_to_patches[layer] = []
        modules_to_patches[layer].append(patch)

    unique_modules = list(modules_to_patches.keys())

    def perform_patch(repr, layer_name):
        # print(layer_name)
        if layer_name not in unique_modules:
            return repr

        current_state = (
            repr
            if "mlp" in layer_name or "embed" in layer_name or "_proj" in layer_name
            else repr[0]
        )
        if current_state.dim() == 2:
            current_state = current_state.unsqueeze(0)
        for patch in modules_to_patches[layer_name]:
            # current_state[:, patch.index, :] = patch.patch
            if len(patch.location) == 2:
                _, token_idx = patch.location
                head_idx = None
            elif len(patch.location) == 3:
                _, head_idx, token_idx = patch.location
            else:
                raise ValueError(f"Invalid patch location: {patch.location}")

            if head_idx is None:
                if patch.strategy == "replace":
                    current_state[:, token_idx, :] = patch.patch
                elif patch.strategy == "add":
                    current_state[:, token_idx, :] += patch.patch
                else:
                    raise ValueError(f"Unknown patch strategy: {patch.strategy}")
            else:
                current_state = current_state.reshape(
                    batch_size, seq_len, n_heads, head_dim
                ).transpose(1, 2)
                if patch.strategy == "replace":
                    current_state[:, head_idx, token_idx, :] = patch.patch
                elif patch.strategy == "add":
                    current_state[:, head_idx, token_idx, :] += patch.patch
                else:
                    raise ValueError(f"Unknown patch strategy: {patch.strategy}")
                current_state = current_state.transpose(1, 2).reshape(
                    batch_size, seq_len, n_heads * head_dim
                )

        return repr

    with baukit.TraceDict(
        module=model, layers=unique_modules, edit_output=perform_patch
    ):
        output = model(**inputs, **model_kwargs)

    return output


def patch_with_nnsight(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    patches: list[PatchSpec] = [],
    model_kwargs: dict = {},
):
    batch_size = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]
    n_heads = mt.config.num_attention_heads
    head_dim = mt.config.hidden_size // n_heads

    with mt.trace(inputs, **model_kwargs) as tracer:
        for cur_patch in patches:
            if len(cur_patch.location) == 2:
                module_name, index = cur_patch.location
                head_idx = None
            elif len(cur_patch.location) == 3:
                module_name, head_idx, index = cur_patch.location

            print(module_name, index, head_idx, cur_patch.strategy)
            module = get_module_nnsight(mt, module_name)
            current_state = (
                module.output
                if (
                    "mlp" in module_name
                    or module_name == mt.embedder_name
                    or "_proj" in module_name
                )
                else module.output[0]
            )
            if current_state.ndim == 2:
                current_state = current_state.unsqueeze(0)

            # current_state = current_state.clone()
            tracer.log(
                "log",
                current_state.shape,
                cur_patch.location,
                cur_patch.patch.shape,
            )
            if head_idx is None:
                print("is working")
                tracer.log(">> false")
                if cur_patch.strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )
                # module.output[...] = current_state

            else:
                print("-- not working")
                tracer.log(">> tracer")
                # print("HI", cur_patch.strategy, cur_patch.location)
                current_state = current_state.clone()
                current_state = current_state.reshape(
                    batch_size, seq_len, n_heads, head_dim
                ).transpose(1, 2)
                tracer.log(current_state.shape, head_idx, index)

                if cur_patch.strategy == "replace":
                    current_state[:, head_idx, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, head_idx, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )
                current_state = current_state.transpose(1, 2).reshape(
                    batch_size, seq_len, n_heads * head_dim
                )
                tracer.log(current_state.shape)
                print(current_state.shape, "Setting")
                module.output[...] = current_state

        output = mt.output.save()
    return output


@torch.inference_mode()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> torch.Tensor | dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]
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

    layer_names = [layer_name for layer_name, _ in locations]
    layer_names = list(set(layer_names))
    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with mt.trace(input, scan=False):
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
                if current_state.ndim == 2:
                    current_state = current_state.unsqueeze(0)
                if cur_patch.strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif cur_patch.strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError(
                        f"patch_strategy must be one of 'replace', 'add'. got {cur_patch.strategy}"
                    )

        for layer_name in layer_names:
            if is_an_attn_head(layer_name) is False:
                module = get_module_nnsight(mt, layer_name)
                layer_states[layer_name] = module.output.save()
            else:
                attn_module_name, head_idx = is_an_attn_head(layer_name)
                o_proj_name = attn_module_name + ".o_proj"
                head_dim = mt.n_embd // mt.model.config.num_attention_heads
                o_proj = get_module_nnsight(mt, o_proj_name)
                layer_states[layer_name] = o_proj.input[0][0][
                    :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                ].save()

    hs = {}

    for layer_name, index in locations:
        # print(
        #     layer_name, layer_states[layer_name].shape, type(layer_states[layer_name])
        # )
        hs[(layer_name, index)] = untuple(layer_states[layer_name].value)[
            :, index, :
        ].squeeze()

    # print(f"==========> {len(hs)=}")
    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


def extract_rep_at_pos(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    total_length: int,
    locations: tuple[str, int] | list[tuple[str, int]],
    return_dict: bool = False,
):
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    assert total_length >= len(
        input["input_ids"][0]
    ), "Total length cannot be smaller than the input length"

    input = insert_padding_before_pos(
        inp=input,
        token_position=0,  # prepend pads before the first token
        pad_len=total_length - len(input["input_ids"][0]),
        pad_id=mt.tokenizer.pad_token_id,
    )

    logger.debug(f"{input.input_ids.shape=}")

    return get_hs(mt=mt, input=input, locations=locations, return_dict=return_dict)


@torch.inference_mode
def get_all_module_states(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError("kind must be one of 'residual', 'mlp', 'attention'")

    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


#! Obsolete code related to the Bridge dataset. Should be moved to their own file if required in the future.
# def predict_bridge_entity(
#     mt: ModelandTokenizer,
#     prompt: str,
#     search_span: int = 100,
#     separator: Literal["#", "-"] = "#",
# ) -> str:
#     inputs = prepare_input(prompts=prompt, tokenizer=mt, add_bos_token=False)

#     bridge_entity = ""
#     while search_span > 0:
#         predicted_token = predict_next_token(mt=mt, inputs=inputs, k=1)[0][0]
#         if predicted_token.token.strip() == separator:
#             break

#         bridge_entity += predicted_token.token
#         # print(torch.tensor([predicted_token.token_id]).to(mt.device))
#         # print(inputs["input_ids"].device)
#         inputs["input_ids"] = torch.cat(
#             [
#                 inputs["input_ids"],
#                 torch.tensor([predicted_token.token_id])[None].to(mt.device),
#             ],
#             dim=1,
#         )
#         inputs["attention_mask"] = torch.cat(
#             [inputs["attention_mask"], torch.tensor([1])[None].to(mt.device)], dim=1
#         )
#         search_span -= 1

#         if search_span == 0:
#             logger.error(f"search span exceeded - found: {bridge_entity}")
#             return bridge_entity

#     return bridge_entity


# def verify_bridge_response(
#     query_sample: BridgeSample,
#     predicted_answer: str,
#     model: str = "claude",
# ) -> str:
#     prompt = f"""
# A smaller language model was asked the following question:
# "What is a common link between {query_sample.entity_pair[0]} and {query_sample.entity_pair[1]}?"
# And the model gave the following answer:
# "{predicted_answer.strip()}"
# Is it correct? Your answer should start with "Yes" or "No". If the answer is "Yes", don't say anything else. If the answer is "No", give explanation why.
# """
#     return ASK_ORACLE_MODEL[model](prompt)


# @torch.inference_mode()
# def filter_bridge_samples_by_model_knowledge(
#     mt: ModelandTokenizer,
#     dataset: BridgeDataset,
#     limit: Optional[int] = None,
#     powerful_LM: str = "claude",
# ) -> BridgeDataset:
#     filtered_samples = []
#     filtered_relation_samples = {}
#     for i in tqdm(range(len(dataset))):
#         prompt, answer = dataset[i]
#         sample = dataset.examples[i]
#         predicted_bridge = predict_bridge_entity(mt, prompt)
#         # is_correct = is_nontrivial_prefix(sample.bridge.lower(), predicted_bridge) or is_nontrivial_prefix(predicted_bridge, sample.bridge)
#         is_correct = (
#             sample.bridge.strip().lower() == "none"
#             and predicted_bridge.strip().lower().startswith("none")
#         ) or (
#             verify_bridge_response(sample, predicted_bridge, powerful_LM)
#             .lower()
#             .startswith("yes")
#         )

#         logger.info(
#             f"{sample.entity_pair} <> {sample.bridge} | predicted: {predicted_bridge} => ({get_tick_marker(is_correct)})"
#         )
#         if is_correct:
#             filtered_samples.append(sample)
#             if sample.relation not in filtered_relation_samples:
#                 filtered_relation_samples[sample.relation] = []
#             filtered_relation_samples[sample.relation].append(sample)
#         if limit is not None and len(filtered_samples) >= limit:
#             break

#     logger.info(
#         f"filtered {len(filtered_samples)} samples out of {len(dataset)} with {len(dataset.icl_examples)} icl examples"
#     )

#     dataset.examples = filtered_samples
#     for relation in dataset.relations:
#         relation.examples = filtered_relation_samples.get(relation.name, [])

#     return dataset


def free_gpu_cache():
    # before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    # after = torch.cuda.memory_allocated()
    # freed = before - after

    # logger.debug(
    #     f"freed {models.bytes_to_human_readable(freed)} | before={models.bytes_to_human_readable(before)} -> after={models.bytes_to_human_readable(after)}"
    # )


def get_dummy_input(
    tokenizer: ModelandTokenizer | Tokenizer,
):
    dummy_prompt = "The quick brown fox"
    return prepare_input(prompts=dummy_prompt, tokenizer=tokenizer)


def low_rank_approx(
    matrix: torch.Tensor, rank: int, svd: SVD | None = None
) -> torch.Tensor:
    """Compute a low-rank approximation of a matrix.

    Args:
        matrix: The matrix to approximate.
        rank: The rank of the approximation.

    Returns:
        The approximation.

    """
    if svd is None:
        svd = SVD.calculate(matrix.float())
    u, s, v = svd.U, svd.S, svd.Vh
    matrix_approx = u[:, :rank] @ torch.diag(s[:rank]) @ v[:, :rank].T
    return matrix_approx.to(matrix.dtype)


def low_rank_pinv(
    matrix: torch.Tensor, rank: int, svd: SVD | None = None
) -> torch.Tensor:
    """Compute a low-rank pseudo-inverse of a matrix.

    Args:
        matrix: The matrix to invert.
        rank: The rank of the approximation.

    Returns:
        The pseudo-inverse.

    """
    if svd is None:
        svd = SVD.calculate(matrix.float())
    u, s, v = svd.U, svd.S, svd.Vh
    matrix_pinv = v[:, :rank] @ torch.diag(1 / s[:rank]) @ u[:, :rank].T
    return matrix_pinv.to(matrix.dtype)


# useful for saving with jsons
def detensorize(inp: dict[Any, Any] | list[dict[Any, Any]], to_numpy: bool = False):
    if isinstance(inp, list):
        return [detensorize(i) for i in inp]
    if isinstance(inp, dict) is False:
        try:
            cls = type(inp)
            inp = inp.__dict__
        except Exception:
            return inp
    else:
        cls = None

    inp = copy.deepcopy(inp)
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            if len(inp[k].shape) == 0:
                inp[k] = inp[k].item()
            else:
                inp[k] = inp[k].tolist() if to_numpy is False else inp[k].cpu().numpy()
        else:
            inp[k] = detensorize(inp[k])

    free_gpu_cache()

    if cls is None:
        return inp
    else:
        if cls != TokenizerOutput:
            return cls(**inp)
        else:
            return cls(data=inp)


def save_object(obj, save_path):
    """
    saves an object as a npz file
    assume that the obj is a dictionary
    and does not have another custom object as a value
    """
    if isinstance(obj, dict) is False:
        obj = detensorize(obj)
    np.savez(
        save_path,
        **obj,
    )

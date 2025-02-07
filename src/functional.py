import copy
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm

from src.dataset import BridgeDataset, BridgeSample, Relation
from src.models import ModelandTokenizer, is_llama_variant
from src.tokens import find_token_range, prepare_input
from src.utils.env_utils import CLAUDE_CACHE_DIR, GPT_4O_CACHE_DIR
from src.utils.typing import ArrayLike, PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
) -> list[PredictedToken]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()

    return [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: list[int] = [],
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    with mt.trace(get_dummy_input(mt), scan=True, validate=True) as trace:
        lnf = get_module_nnsight(mt, mt.final_layer_norm_name)
        lnf.input = h.view(1, 1, h.squeeze().shape[0])
        logits = mt.output.logits.save()

    logits = logits.squeeze()
    candidates = interpret_logits(tokenizer=mt, logits=logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=mt.tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        return candidates, interested_logits
    free_gpu_cache()
    return candidates


@torch.inference_mode()
def forward_pass_to_vocab(
    mt: ModelandTokenizer, h: torch.Tensor, layer_name: str, **kwargs
):
    inputs = mt.tokenizer(
        mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    )
    with mt.trace(inputs) as tr:
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
    layer_idx: 10,
    interested_tokens: list[int] = [],
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    placeholder = "placeholder"
    copy_prompt = f"cat -> cat; hello -> hello; Microsoft -> Microsoft; copy -> copy; Python -> Python; {placeholder} ->"
    input = prepare_input(
        tokenizer=mt,
        prompts=copy_prompt,
        return_offset_mapping=True,
    )
    placeholder_range = find_token_range(
        string=copy_prompt,
        substring=placeholder,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=input["offset_mapping"][0],
    )
    placeholder_pos = placeholder_range[1] - 1
    input.pop("offset_mapping")
    logger.debug(
        f"placeholder position: {placeholder_pos} | token: {mt.tokenizer.decode(input['input_ids'][0, placeholder_pos])}"
    )

    processed_h = get_hs(
        mt=mt,
        input=input,
        locations=[(mt.layer_names[-1], -1)],
        patches=PatchSpec(
            location=(mt.layer_name_format.format(layer_idx), placeholder_pos),
            patch=h,
        ),
        return_dict=False,
    )
    return logit_lens(
        mt=mt,
        h=processed_h,
        interested_tokens=interested_tokens,
        k=k,
    )


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


def generate_with_patch(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    n_gen_per_prompt: int = 5,
    max_new_tokens: int = 20,
    patches: Optional[list[PatchSpec]] = None,
    use_kv_cache: bool = True,
    do_sample: bool = True,
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
        use_cache=use_kv_cache,
    ) as gen_trace:
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                current_state[:, index, :] = cur_patch.patch
        gen_out = mt.generator.output.save()

    return mt.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)


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
        print(f"{len(token_of_interest)=} | {len(inputs)=}")
        assert len(token_of_interest) == (
            len(inputs["input_ids"])
            if isinstance(inputs, TokenizerOutput)
            else len(inputs)
        )
        track_interesting_tokens = []

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]
        logger.warning(
            "passed `patches`, not supported for batched predictions yet. will give weird results."
        )

    predictions = []
    is_tokenized = isinstance(inputs, TokenizerOutput)
    total_len = len(inputs["input_ids"]) if is_tokenized else len(inputs)
    for i in range(0, total_len, batch_size):
        if is_tokenized == False:
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

        with mt.trace(batch_inputs, scan=False, validate=False) as tr:
            # TODO: patching code is being repeated a couple of times. refactor it.
            if patches is not None:
                for cur_patch in patches:
                    module_name, index = cur_patch.location
                    if is_an_attn_head(module_name) != False:
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

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            predictions.append(
                [
                    PredictedToken(
                        token=mt.tokenizer.decode(token_ids[j]),
                        prob=token_probs[j].item(),
                        logit=batch_logits[batch_order][token_ids[j]].item(),
                        token_id=token_ids[j].item(),
                    )
                    for j in range(k)
                ]
            )

        if token_of_interest is not None:
            for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                tok_id = (
                    mt.tokenizer(
                        token_of_interest[j], add_special_tokens=False
                    ).input_ids[0]
                    if type(token_of_interest[j]) == str
                    else token_of_interest[j]
                )
                probs = batch_probs[j]
                order = probs.topk(dim=-1, k=probs.shape[-1]).indices.squeeze()
                prob_tok = probs[tok_id]
                rank = int(torch.where(order == tok_id)[0].item() + 1)
                track_interesting_tokens.append(
                    (
                        rank,
                        PredictedToken(
                            token=mt.tokenizer.decode(tok_id),
                            prob=prob_tok.item(),
                            logit=batch_logits[j - i][tok_id].item(),
                            token_id=tok_id,
                        ),
                    )
                )

        free_gpu_cache()

    if token_of_interest is not None:
        return predictions, track_interesting_tokens

    return predictions


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


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
    with mt.trace(input, scan=False) as tracer:
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) != False:
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

        for layer_name in layer_names:
            if is_an_attn_head(layer_name) == False:
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
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def predict_bridge_entity(
    mt: ModelandTokenizer,
    prompt: str,
    search_span: int = 100,
    separator: Literal["#", "-"] = "#",
) -> str:
    inputs = prepare_input(prompts=prompt, tokenizer=mt, add_bos_token=False)

    bridge_entity = ""
    while search_span > 0:
        predicted_token = predict_next_token(mt=mt, inputs=inputs, k=1)[0][0]
        if predicted_token.token.strip() == separator:
            break

        bridge_entity += predicted_token.token
        # print(torch.tensor([predicted_token.token_id]).to(mt.device))
        # print(inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat(
            [
                inputs["input_ids"],
                torch.tensor([predicted_token.token_id])[None].to(mt.device),
            ],
            dim=1,
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.tensor([1])[None].to(mt.device)], dim=1
        )
        search_span -= 1

        if search_span == 0:
            logger.error(f"search span exceeded - found: {bridge_entity}")
            return bridge_entity

    return bridge_entity


def ask_gpt4o(
    prompt: str,
) -> str:
    ##################################################
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )
    MODEL_NAME = "gpt-4o"
    ##################################################

    hash_val = hashlib.md5(prompt.encode()).hexdigest()
    if f"{hash_val}.json" in os.listdir(GPT_4O_CACHE_DIR):
        logger.debug(f"found cached gpt4o response for {hash_val} - loading")
        with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "r") as f:
            json_data = json.load(f)
            return json_data["response"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4000,
    )
    response = response.choices[0].message.content

    with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "tempraure": 0,
            },
            f,
        )

    return response


def ask_claude(
    prompt: str,
) -> str:
    ##################################################
    client = Anthropic(
        api_key=os.getenv("CLAUDE_KEY"),
    )
    MODEL_NAME = "claude-3-5-sonnet-20241022"
    ##################################################

    hash_val = hashlib.md5(prompt.encode()).hexdigest()
    if f"{hash_val}.json" in os.listdir(CLAUDE_CACHE_DIR):
        logger.debug(f"found cached gpt4o response for {hash_val} - loading")
        with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "r") as f:
            json_data = json.load(f)
            return json_data["response"]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        temperature=0,
        system="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )
    response = response.content[0].text

    with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "tempraure": 0,
            },
            f,
        )

    return response


ASK_ORACLE_MODEL = {"gpt4o": ask_gpt4o, "claude": ask_claude}


def verify_bridge_response(
    query_sample: BridgeSample,
    predicted_answer: str,
    model: str = "claude",
) -> str:
    prompt = f"""
A smaller language model was asked the following question:
"What is a common link between {query_sample.entity_pair[0]} and {query_sample.entity_pair[1]}?"
And the model gave the following answer:
"{predicted_answer.strip()}"
Is it correct? Your answer should start with "Yes" or "No". If the answer is "Yes", don't say anything else. If the answer is "No", give explanation why.
"""
    return ASK_ORACLE_MODEL[model](prompt)


@torch.inference_mode()
def filter_bridge_samples_by_model_knowledge(
    mt: ModelandTokenizer,
    dataset: BridgeDataset,
    limit: Optional[int] = None,
    powerful_LM: str = "claude",
) -> BridgeDataset:
    filtered_samples = []
    filtered_relation_samples = {}
    for i in tqdm(range(len(dataset))):
        prompt, answer = dataset[i]
        sample = dataset.examples[i]
        predicted_bridge = predict_bridge_entity(mt, prompt)
        # is_correct = is_nontrivial_prefix(sample.bridge.lower(), predicted_bridge) or is_nontrivial_prefix(predicted_bridge, sample.bridge)
        is_correct = (
            sample.bridge.strip().lower() == "none"
            and predicted_bridge.strip().lower().startswith("none")
        ) or (
            verify_bridge_response(sample, predicted_bridge, powerful_LM)
            .lower()
            .startswith("yes")
        )

        logger.info(
            f"{sample.entity_pair} <> {sample.bridge} | predicted: {predicted_bridge} => ({get_tick_marker(is_correct)})"
        )
        if is_correct:
            filtered_samples.append(sample)
            if sample.relation not in filtered_relation_samples:
                filtered_relation_samples[sample.relation] = []
            filtered_relation_samples[sample.relation].append(sample)
        if limit is not None and len(filtered_samples) >= limit:
            break

    logger.info(
        f"filtered {len(filtered_samples)} samples out of {len(dataset)} with {len(dataset.icl_examples)} icl examples"
    )

    dataset.examples = filtered_samples
    for relation in dataset.relations:
        relation.examples = filtered_relation_samples.get(relation.name, [])

    return dataset


def free_gpu_cache():
    before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    freed = before - after

    # logger.debug(
    #     f"freed {models.bytes_to_human_readable(freed)} | before={models.bytes_to_human_readable(before)} -> after={models.bytes_to_human_readable(after)}"
    # )


def get_dummy_input(
    tokenizer: ModelandTokenizer | Tokenizer,
):
    dummy_prompt = "The quick brown fox"
    return prepare_input(prompts=dummy_prompt, tokenizer=tokenizer)


# useful for saving with jsons
def detensorize(inp: dict[Any, Any] | list[dict[Any, Any]], to_numpy: bool = False):
    if isinstance(inp, list):
        return [detensorize(i) for i in inp]
    if isinstance(inp, dict) == False:
        try:
            cls = type(inp)
            inp = inp.__dict__
        except:
            return inp
    else:
        cls = None

    inp = copy.deepcopy(inp)
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            if len(inp[k].shape) == 0:
                inp[k] = inp[k].item()
            else:
                inp[k] = inp[k].tolist() if to_numpy == False else inp[k].cpu().numpy()
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

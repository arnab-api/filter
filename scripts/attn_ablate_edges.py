import argparse
import itertools
import json
import logging
import os
import types
from dataclasses import dataclass
from typing import Literal, Optional

import baukit
import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.dataset import BridgeDataset, BridgeRelation
from src.functional import predict_next_token
from src.hooking.llama_attention import AttentionEdge, LlamaAttentionPatcher
from src.models import ModelandTokenizer
from src.tokens import find_token_range, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PredictedToken

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadAblationSweepResult(DataClassJsonMixin):
    question: str
    entity_pair: tuple[str, str]
    answer: PredictedToken
    tokens: list[str]
    block_edges: list[AttentionEdge]
    effects: dict[int, PredictedToken]


def do_attn_blocking_sweep(
    mt: ModelandTokenizer,
    question: str,
    entity_pair: tuple[str, str],
    focus_strategy: Literal["subj", "subj_last", "ablate_all"] = "subj",
):
    input = prepare_input(prompts=question, tokenizer=mt, return_offset_mapping=True)
    subj1_range = find_token_range(
        string=question,
        substring=entity_pair[0],
        offset_mapping=input.offset_mapping[0],
        tokenizer=mt.tokenizer,
    )
    subj1_last = subj1_range[-1] - 1
    logger.info(
        f"{subj1_last=} | {mt.tokenizer.decode(input.input_ids[0][subj1_last])=}"
    )

    subj2_range = find_token_range(
        string=question,
        substring=entity_pair[1],
        offset_mapping=input.offset_mapping[0],
        tokenizer=mt.tokenizer,
    )
    subj2_last = subj2_range[-1] - 1
    logger.info(
        f"{subj2_last=} | {mt.tokenizer.decode(input.input_ids[0][subj2_last])=}"
    )

    mt.reset_forward()
    answer = predict_next_token(mt=mt, inputs=input, k=1)[0][0]
    logger.info(f"{answer=}")

    if focus_strategy == "subj":
        whitelist_k_indices = (
            [0] + list(range(*subj1_range)) + list(range(*subj2_range))
        )
    elif focus_strategy == "subj_last":
        whitelist_k_indices = [0] + [subj1_last, subj2_last]
    elif focus_strategy == "ablate_all":
        whitelist_k_indices = [0]
    else:
        raise ValueError(f"Unknown focus_strategy: {focus_strategy}")

    block_edges: list[AttentionEdge] = []

    Q_IDX = -1
    for k_idx in range(1, input.input_ids[0].shape[-1]):
        if k_idx in whitelist_k_indices:
            continue
        block_edges.append(
            AttentionEdge(
                q_idx=Q_IDX,
                k_idx=k_idx,
            )
        )

    block_for_all_attn_heads = {
        h_idx: block_edges for h_idx in range(mt.config.num_attention_heads)
    }
    effects = {}
    for l in range(mt.n_layer - 1, -1, -1):
        attn_block_name = mt.attn_module_name_format.format(l)
        attn_block = baukit.get_module(mt._model, attn_block_name)
        attn_block.forward = types.MethodType(
            LlamaAttentionPatcher(
                block_name=attn_block_name,
                cut_attn_edges=block_for_all_attn_heads,
            ),
            attn_block,
        )
        ablated_pred, track_ans = predict_next_token(
            mt=mt, inputs=input, k=1, token_of_interest=[answer.token_id]
        )
        rank, track_ans = list(track_ans)[0]
        logger.debug(f"{l=} >> {rank=} | {track_ans=}")
        effects[l] = track_ans

    mt.reset_forward()
    return HeadAblationSweepResult(
        question=question,
        entity_pair=entity_pair,
        answer=answer,
        tokens=[mt.tokenizer.decode(t) for t in input.input_ids[0]],
        block_edges=block_edges,
        effects=effects,
    )


@dataclass(frozen=False)
class ExperimentResults(DataClassJsonMixin):
    model_name: str
    relation_name: str
    focus_strategy: Literal["subj", "subj_last", "ablate_all"]
    sweep_results: list[HeadAblationSweepResult]


def run_experiment(
    model_name: str,
    relation: str,
    known_data_file: str,
    save_dir: Optional[str] = None,
    limit: Optional[int] = None,
    focus_strategy: Literal["subj", "subj_last", "ablate_all"] = "subj",
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )
    if save_dir is not None:
        save_dir = os.path.join(
            env_utils.DEFAULT_RESULTS_DIR,
            save_dir,
            mt.name.split("/")[-1],
            relation,
            focus_strategy,
        )
        os.makedirs(save_dir, exist_ok=True)

    cached_known_dir = os.path.join(
        env_utils.DEFAULT_DATA_DIR, "bridge_cached", mt.name.split("/")[-1]
    )
    with open(os.path.join(cached_known_dir, known_data_file), "r") as f:
        json_data = json.load(f)

    dataset = BridgeDataset.from_dict(json_data)
    if relation != "all":
        relation_icq = None
        for rel in json_data["relations"]:
            if rel["name"] == relation:
                relation_icq = BridgeRelation.from_dict(rel)
                break
        assert (
            relation_icq is not None
        ), f"{relation=} is not found. Available relations: {[r['name'] for r in json_data['relations']]}"
        dataset.examples = relation_icq.examples
    dataset.ensure_icl_not_in_examples()

    logger.debug(f"{dataset.icl_examples=}")

    experiment_results = ExperimentResults(
        model_name=model_name,
        relation_name=relation,
        focus_strategy=focus_strategy,
        sweep_results=[],
    )
    limit = len(dataset) if limit is None else limit
    for idx in range(limit):
        logger.info(f"Processing {idx+1}/{limit}")
        question, answer = dataset[idx]
        logger.debug(f"{question=}")
        entity_pair = dataset.examples[idx].entity_pair

        experiment_results.sweep_results.append(
            do_attn_blocking_sweep(
                mt=mt,
                question=question,
                entity_pair=entity_pair,
                focus_strategy=focus_strategy,
            )
        )

        if save_dir is not None:
            with open(os.path.join(save_dir, f"{relation}.json"), "w") as f:
                f.write(experiment_results.to_json(indent=2))

    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        choices=["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--relation",
        type=str,
        choices=[
            "architect_building",
            "movie_actor",
            "sport_players",
            "superpower_characters",
            "none",
            "all",
        ],
        default="sport_players",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ablate_edges",
    )
    parser.add_argument(
        "--focus",
        type=str,
        choices=["subj", "subj_last", "ablate_all"],
        default="subj",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--known_data",
        type=str,
        default="filtered_2024-09-06T00:58:32.777195.json",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)
    logger.info(f"{args=}")

    kwargs = dict(
        model_name=args.model,
        relation=args.relation,
        known_data_file=args.known_data,
        save_dir=args.save_dir,
        focus_strategy=args.focus,
        limit=args.limit if args.limit > 0 else None,
    )

    run_experiment(**kwargs)

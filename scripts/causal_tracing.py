import argparse
import itertools
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

from src.dataset import BridgeDataset, BridgeRelation, BridgeSample
from src.functional import get_hs, predict_next_token
from src.models import ModelandTokenizer
from src.tokens import align_bridge_entities_in_query
from src.trace import calculate_indirect_effects
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import PredictedToken

logger = logging.getLogger(__name__)


def sample_icq_with_different_bridge(
    icq_examples: list[BridgeSample], patch_icq: BridgeSample
) -> int:
    indices = np.random.permutation(len(icq_examples))
    for idx in indices:
        if icq_examples[idx].bridge != patch_icq.bridge:
            return idx
    raise ValueError("No sample found with a different bridge")


@dataclass(frozen=True)
class CacheCausalTracingResult(DataClassJsonMixin):
    clean_ques: str
    clean_toks: list[str]
    clean_ans: PredictedToken

    patch_ques: str  # h patched from. The fact we care about
    patch_toks: list[str]
    patch_ans: PredictedToken
    corrupt_patch_ans: tuple[int, PredictedToken]

    query_start: int
    subj_1_range: tuple[int, int]
    subj_2_range: tuple[int, int]
    indirect_effects: dict[str, float]


def get_causal_tracing_results_for_bridge_pair(
    mt: ModelandTokenizer,
    clean_ques: str,
    clean_entity_pair: tuple[str, str],
    patch_ques: str,
    patch_entity_pair: tuple[str, str],
    kind_window_size: dict[str, int],
):
    inps = align_bridge_entities_in_query(
        mt=mt,
        clean_ques=clean_ques,
        clean_entity_pair=clean_entity_pair,
        patch_ques=patch_ques,
        patch_entity_pair=patch_entity_pair,
    )
    clean_inputs = inps["clean_inputs"]
    patch_inputs = inps["patch_inputs"]
    query_start = inps["query_start"]
    subj_1_range = inps["subj_1_range"]
    subj_2_range = inps["subj_2_range"]

    patch_ans = predict_next_token(
        mt=mt,
        inputs=patch_inputs,
    )[
        0
    ][0]
    logger.debug(f"{patch_ans=}")

    clean_ans, corrupt_rank = predict_next_token(
        mt=mt,
        inputs=clean_inputs,
        token_of_interest=[mt.tokenizer.decode(patch_ans.token_id)],
    )
    clean_ans = clean_ans[0][0]
    corrupt_rank, base_ans = corrupt_rank[0]
    logger.debug(f"{clean_ans=}")
    logger.debug(f"{corrupt_rank=} | {base_ans=}")

    assert clean_inputs["input_ids"].shape[1] == patch_inputs["input_ids"].shape[1]

    locations = list(
        itertools.product(
            range(mt.n_layer),
            range(query_start, clean_inputs.input_ids.shape[1]),
        )
    )

    kind_to_layer_name_format = {
        "residual": mt.layer_name_format,
        "attention": mt.attn_module_name_format,
        "mlp": mt.mlp_module_name_format,
    }

    indirect_effects = {}

    for kind in kind_window_size:
        layer_name_format = kind_to_layer_name_format[kind]
        window_size = kind_window_size[kind]
        patch_states = get_hs(
            mt=mt,
            input=patch_inputs,
            locations=[(layer_name_format.format(l), t) for l, t in locations],
            return_dict=True,
        )

        kind_results = calculate_indirect_effects(
            mt=mt,
            locations=locations,
            clean_input=clean_inputs,
            patch_states=patch_states,
            patch_ans_t=patch_ans.token_id,
            layer_name_format=layer_name_format,
            window_size=window_size,
        )

        for (layer_idx, tok_idx), val in kind_results.items():
            indirect_effects[f"{layer_name_format.format(layer_idx)}_<>_{tok_idx}"] = (
                val
            )

    return CacheCausalTracingResult(
        clean_ques=clean_ques,
        clean_toks=[mt.tokenizer.decode(t) for t in clean_inputs["input_ids"][0]],
        clean_ans=clean_ans,
        patch_ques=patch_ques,
        patch_toks=[mt.tokenizer.decode(t) for t in patch_inputs["input_ids"][0]],
        patch_ans=patch_ans,
        corrupt_patch_ans=(corrupt_rank, base_ans),
        query_start=query_start,
        subj_1_range=subj_1_range,
        subj_2_range=subj_2_range,
        indirect_effects=indirect_effects,
    )


@dataclass(frozen=False)
class ExperimentResults(DataClassJsonMixin):
    model_name: str
    relation_name: str
    window_size: dict[str, int]
    causal_tracing_results: list[CacheCausalTracingResult]


def cache_causal_tracing_results(
    model_name: str,
    relation: str,
    known_data_file: str,
    save_dir: Optional[str] = None,
    limit: Optional[int] = None,
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
        model_name=mt.name,
        window_size={
            "residual": 1,
            "attention": 5,
            "mlp": 5,
        },
        relation_name=relation,
        causal_tracing_results=[],
    )
    limit = len(dataset) if limit is None else limit
    for idx in range(limit):
        logger.info(f"Processing {idx+1}/{limit}")
        patch_qa = dataset[idx]
        clean_idx = sample_icq_with_different_bridge(
            dataset.examples, dataset.examples[idx]
        )
        clean_qa = dataset[clean_idx]

        logger.debug(f"{clean_qa=}")
        logger.debug(f"{patch_qa=}")

        atp_result = get_causal_tracing_results_for_bridge_pair(
            mt=mt,
            clean_ques=clean_qa[0],
            clean_entity_pair=dataset.examples[clean_idx].entity_pair,
            patch_ques=patch_qa[0],
            patch_entity_pair=dataset.examples[idx].entity_pair,
            kind_window_size=experiment_results.window_size,
        )

        experiment_results.causal_tracing_results.append(atp_result)

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
        default="causal_tracing",
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
        limit=args.limit if args.limit > 0 else None,
    )

    cache_causal_tracing_results(**kwargs)

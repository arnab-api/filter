import argparse
import json
import logging
import os
import types
from dataclasses import dataclass
from typing import Optional

import baukit  # type: ignore
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

from src.dataset import BridgeDataset, BridgeRelation
from src.functional import predict_next_token
from src.hooking.llama_attention import AttentionEdge, LlamaAttentionPatcher
from src.models import ModelandTokenizer, prepare_input
from src.utils import env_utils, logging_utils, experiment_utils
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadAblationEffect(DataClassJsonMixin):
    layer: int
    head: int
    effect: float


@dataclass(frozen=True)
class AblationEffects(DataClassJsonMixin):
    question: str
    effects: list[list[float]]


@dataclass(frozen=False)
class ExperimentResults(DataClassJsonMixin):
    model_name: str
    ablation_effects: list[AblationEffects]
    top_heads: list[HeadAblationEffect]


def get_ablation_effect_of_heads(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    heads: dict[int, list[int]],  # {layer: [head]}
    ablation_spec: list[AttentionEdge],
) -> torch.Tensor:

    mt.reset_forward()

    for l in heads:
        module_name = mt.attn_module_name_format.format(l)
        attn_module = baukit.get_module(mt._model, module_name)
        attn_module.forward = types.MethodType(
            LlamaAttentionPatcher(
                cut_attn_edges={h: ablation_spec for h in heads[l]},
            ),
            attn_module,
        )
    output = mt._model(**inputs)

    mt.reset_forward()
    return output.logits[:, -1, :].squeeze()


def get_ablation_results_for_all_heads(
    mt: ModelandTokenizer,
    question: str,
) -> torch.Tensor:

    mt.reset_forward()

    inputs = prepare_input(prompts=question, tokenizer=mt, add_bos_token=False)
    base_prediction = predict_next_token(mt, inputs, k=1)[0][0]
    head_ablation_results = torch.zeros(mt.n_layer, mt.config.num_attention_heads) - 1

    # q can only attend to <bos>. cut all other edges.
    ablation_spec: list[AttentionEdge] = []
    for q_idx in range(1, inputs["input_ids"].shape[1]):
        ablation_spec.extend(
            [AttentionEdge(q_idx, k_idx) for k_idx in range(1, q_idx + 1)]
        )

    for layer in tqdm(range(mt.n_layer)):
        for head in range(mt.config.num_attention_heads):
            ablated_logits = get_ablation_effect_of_heads(
                mt, inputs, {layer: [head]}, ablation_spec
            )
            head_ablation_results[layer, head] = ablated_logits.softmax(dim=-1)[
                base_prediction.token_id
            ].item()

    mt.reset_forward()

    # normalize
    head_ablation_effects = (
        base_prediction.prob - head_ablation_results
    ) / base_prediction.prob
    return head_ablation_effects


def get_top_heads(
    mt: ModelandTokenizer,
    dataset: BridgeDataset,
    save_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> ExperimentResults:

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    limit = limit or len(dataset)

    results: ExperimentResults = ExperimentResults(
        model_name=mt.name,
        ablation_effects=[],
        top_heads=[],
    )

    for idx in tqdm(range(limit)):
        question, answer = dataset[idx]
        logging.info(dataset.examples[idx])
        effects = (
            get_ablation_results_for_all_heads(mt, question).cpu().numpy().tolist()
        )
        results.ablation_effects.append(
            AblationEffects(question=question, effects=effects)
        )
        if save_dir:
            with open(os.path.join(save_dir, "head_ablation.json"), "w") as f:
                f.write(results.to_json(indent=2))

    mean_effects = torch.tensor([e.effects for e in results.ablation_effects]).mean(
        dim=0
    )
    topk = mean_effects.reshape(mean_effects.shape[0] * mean_effects.shape[1]).topk(100)

    indices = [
        (
            i.item() // mean_effects.shape[1],
            i.item() % mean_effects.shape[1],
        )
        for i in topk.indices
    ]

    results.top_heads = [
        HeadAblationEffect(layer=i, head=j, effect=mean_effects[i, j].item())
        for i, j in indices
    ]

    if save_dir:
        with open(os.path.join(save_dir, "head_ablation.json"), "w") as f:
            f.write(results.to_json(indent=2))

    return results


def run_experiment(
    model_name: str,
    relation: str,
    known_data_file: str,
    save_dir: str,
    limit: Optional[int] = None,
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, save_dir, mt.name.split("/")[-1], relation
    )

    cached_known_dir = os.path.join(
        env_utils.DEFAULT_DATA_DIR, "bridge_cached", mt.name.split("/")[-1]
    )
    with open(os.path.join(cached_known_dir, known_data_file), "r") as f:
        json_data = json.load(f)
    if relation == "all":
        dataset = BridgeDataset.from_dict(json_data)
    else:
        for rel in json_data["relations"]:
            if rel["name"] == relation:
                relation_icq = BridgeRelation.from_dict(rel)
                break
        assert (
            relation_icq is not None
        ), f"{relation=} is not found. Available relations: {[r['name'] for r in json_data['relations']]}"
        dataset = BridgeDataset(relations=[relation_icq])

    logger.debug(f"{dataset.icl_examples=}")
    get_top_heads(mt, dataset, save_dir, limit)


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
        "--known_data",
        type=str,
        default="filtered_2024-08-30T23:00:20.070752.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="top_heads",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--relation",
        type=str,
        choices=[
            "architect_building",
            "movie_actor",
            "sport_players",
            "superpower_characters",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    kwargs = dict(
        model_name=args.model,
        relation=args.relation,
        known_data_file=args.known_data,
        save_dir=args.save_dir,
        limit=args.limit if args.limit > 0 else None,
    )

    run_experiment(**kwargs)

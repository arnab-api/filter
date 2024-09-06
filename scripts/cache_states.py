import argparse
import logging
import os

import numpy as np
import torch
import transformers
from datasets import load_dataset

from src.functional import free_gpu_cache, get_module_nnsight
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


@torch.inference_mode()
def cache_activations(
    model_name: str,
    data_name: str,
    limit: int = 20000,
    context_limit: int = 1024,
    save_dir: str = "cached_states",
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
        mt.name.split("/")[-1],
        data_name.split("/")[-1],
    )
    os.makedirs(cache_dir, exist_ok=True)

    if data_name != "wikimedia/wikipedia":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

    context_limit = 1024
    limit = min(limit, len(dataset["train"]))

    all_layers = (
        [mt.embedder_name]  # embeddings
        + mt.layer_names  # residual
        + [
            mt.mlp_module_name_format.format(i) for i in range(mt.n_layer)
        ]  # mlp outputs
        + [
            mt.attn_module_name_format.format(i) for i in range(mt.n_layer)
        ]  # attn outputs
    )

    for doc_index, doc in enumerate(dataset["train"][:limit]["text"]):
        inputs = prepare_input(prompts=doc, tokenizer=mt)
        if inputs["input_ids"].shape[1] > context_limit:
            inputs["input_ids"] = inputs["input_ids"][:, :context_limit]
            inputs["attention_mask"] = inputs["attention_mask"][:, :context_limit]

        # print(f"{doc=}")
        # logger.info(inputs["input_ids"].shape)
        cache = {
            "doc": doc,
            "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int32),
            "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int32),
            "outputs": {layer: None for layer in all_layers},
        }
        with mt.trace(inputs, scan=False, validate=False) as trace:
            for layer_name in all_layers:
                module = get_module_nnsight(mt, layer_name)
                cache["outputs"][layer_name] = (
                    module.output.save()
                    if ("mlp" in layer_name or layer_name == mt.embedder_name)
                    else module.output[0].save()
                )

        # print(cache["doc"])
        # print(f"{cache['input_ids'].shape=} | {cache['attention_mask'].shape=}")
        for layer_name in all_layers:
            # print(f"{layer_name=} | {cache['outputs'][layer_name].size()}")
            cache["outputs"][layer_name] = (
                cache["outputs"][layer_name].cpu().numpy().astype(np.float16)
            )
        cache_path = os.path.join(cache_dir, f"{doc_index}")
        np.savez_compressed(cache_path, allow_pickle=True, **cache)

        free_gpu_cache()

        logger.info(f"Processed {doc_index+1}/{limit} ({(doc_index+1)/limit:.2%})")


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
        "--data",
        type=str,
        choices=["wikimedia/wikipedia"],
        default="wikimedia/wikipedia",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="cache_states",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    cache_activations(
        model_name=args.model,
        data_name=args.data,
        limit=args.limit,
        save_dir=args.save_dir,
    )

import argparse
import json
import logging
import os
from typing import Literal

import numpy as np
import torch
import transformers

from src.functional import detensorize, free_gpu_cache, get_hs
from src.models import ModelandTokenizer
from src.selection.data import SelectionSample
from src.tokens import find_token_range, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.experiment_utils import set_seed
from src.utils.training_utils import TrainableLM_delta, TrainableLM_LoRA

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


@torch.inference_mode()
def cache_selection_states(
    mt: ModelandTokenizer,
    selection_samples: list[SelectionSample],
    token_position: Literal["last_token", "pivot_last"] = "last_token",
    modules: list[str] | None = None,
    save_dir: str | None = None,
):
    """Cache last token states for selection samples."""

    if save_dir is None:
        cached_states = []

    modules = mt.layer_names if modules is None else modules
    logger.info(
        f"Caching states for {len(selection_samples)} samples, "
        f"token_position={token_position}, modules={modules}, "
        f"save_dir={save_dir}"
    )

    # Cache states for each sample
    for idx, sample in enumerate(selection_samples):
        inputs = prepare_input(
            prompts=sample.prompt, tokenizer=mt, return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0]
        if token_position == "last_token":
            tok_idx = len(inputs["input_ids"][0]) - 1
        elif token_position == "pivot_last":
            pivot_range = find_token_range(
                string=sample.prompt,
                substring=sample.subj,
                offset_mapping=offsets,
                tokenizer=mt,
            )
            tok_idx = pivot_range[1] - 1  # last token of the pivot
            logger.debug(
                f"{sample.subj=}, {pivot_range=}, {tok_idx=} | \"{mt.tokenizer.decode(inputs['input_ids'][0][pivot_range[0]:pivot_range[1]])}\", \"{mt.tokenizer.decode(inputs['input_ids'][0][tok_idx])}\""
            )
        else:
            raise ValueError(f"Unknown token_position: {token_position}")

        sample.detensorize()
        locations = [(layer_name, tok_idx) for layer_name in modules]

        cache = {"sample": sample.to_dict(), "states": {}}
        states = get_hs(
            mt=mt,
            input=inputs,
            locations=locations,
            return_dict=True,
        )

        for (layer_name, tok_idx), state in states.items():
            cache["states"][f"{layer_name}_<>_{tok_idx}"] = (
                state.detach().to(torch.float32).cpu().numpy()
            )

        if save_dir is None:
            cached_states.append(cache)

        cache = detensorize(cache)

        if save_dir is not None:
            cache_path = os.path.join(save_dir, f"sample_{idx:05d}.npz")
            np.savez_compressed(cache_path, **cache, allow_pickle=True)

        # Clean up memory periodically
        if (idx + 1) % 10 == 0:
            free_gpu_cache()

        if (idx + 1) % 100 == 0 or idx == len(selection_samples) - 1:
            logger.info(
                f"Processed sample {idx + 1}/{len(selection_samples)} "
                f"({(idx + 1) / len(selection_samples) * 100:.2f}%)"
            )
    logger.info(f"Cached states saved to {save_dir}")

    if save_dir is None:
        return cached_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache selection states for language models"
    )
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model identifier",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="Number of samples to generate and cache",
    )

    parser.add_argument(
        "--token_pos",
        type=str,
        choices=["last_token", "pivot_last"],
        help="Token position to cache states for (-1 for last token)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="selection/Llama-3.3-70B-Instruct/profession/cached_states",
        help="Directory to save cached states",
    )

    parser.add_argument(
        "--selection_ds_path",
        type=str,
        default="selection/Llama-3.3-70B-Instruct/profession/filtered_samples.json",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")

    # loading the model
    mt = ModelandTokenizer(
        model_key=args.model,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
        device_map="auto",
        # quantization_config = BitsAndBytesConfig(
        #     # load_in_4bit=True
        #     load_in_8bit=True
        # )
    )

    # fusing the trained deltas
    SYNTH_DATASET = "64"
    checkpoint_path = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        "trained_params",
        f"{SYNTH_DATASET}",
        "_full__clamp=0.001",
        args.model.split("/")[-1],
    )
    version = "epoch_1"
    checkpoint_path = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, checkpoint_path, version
    )
    checkpoint_path = os.path.join(checkpoint_path, "trainable_params.pt")
    loaded_deltas = torch.load(checkpoint_path, map_location="cpu")
    free_gpu_cache()
    TrainableLM_delta.fuse_with_model(mt._model, loaded_deltas)

    # locations
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

    # load selection samples
    selection_ds_path = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.selection_ds_path
    )
    with open(selection_ds_path, "r") as f:
        selection_samples = [
            SelectionSample.from_dict(d) for d in json.load(f)[: args.limit]
        ]

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        args.token_pos,
    )
    os.makedirs(save_dir, exist_ok=True)

    cache_selection_states(
        mt=mt,
        selection_samples=selection_samples,
        token_position=args.token_pos,
        modules=all_layers,
        save_dir=save_dir,
    )

import argparse
import logging
import os
import random
from typing import Literal

import numpy as np
import torch
import transformers

from src.functional import detensorize, free_gpu_cache, get_hs
from src.models import ModelandTokenizer
from src.selection.data import SelectOddOneOutTask, SelectOneTask
from src.tokens import find_token_range, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


@torch.inference_mode()
def cache_selection_states(
    mt: ModelandTokenizer,
    token_position: Literal["last_token", "pivot_last"] = "last_token",
    modules: list[str] | None = None,
    save_dir: str | None = None,
    task_name: str = "select_one",
    category_type: str = "objects",
    limit: int = 100,
):
    """Cache last token states for selection samples."""

    if save_dir is None:
        cached_states = []

    modules = mt.layer_names if modules is None else modules
    logger.info(
        f"Caching states for selection task={task_name}, category={category_type}, "
        f"token_position={token_position}, modules={modules}, "
        f"save_dir={save_dir}"
    )

    task_cls = {
        "select_one": SelectOneTask,
        "select_odd_one_out": SelectOddOneOutTask,
    }

    select_one_task = task_cls[task_name].load(
        path=os.path.join(
            env_utils.DEFAULT_DATA_DIR, "selection", f"{category_type}.json"
        )
    )

    # Cache states for each sample
    counter = 0
    while counter < limit:
        counter += 1
        n_distractors = random.choice(range(2, 7))
        sample = select_one_task.get_random_sample(
            mt=mt,
            prompt_template_idx=3,
            n_distractors=n_distractors,
            filter_by_lm_prediction=True,
        )
        inputs = prepare_input(
            prompts=sample.prompt(), tokenizer=mt, return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0]
        if token_position == "last_token":
            tok_idx = len(inputs["input_ids"][0]) - 1
        elif token_position == "pivot_last":
            pivot_range = find_token_range(
                string=sample.prompt(),
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
            cache_path = os.path.join(save_dir, f"sample_{counter:05d}.npz")
            np.savez_compressed(cache_path, **cache, allow_pickle=True)

        if counter % 100 == 0 or counter == limit - 1:
            free_gpu_cache()
            logger.info(
                f"Processed sample {counter}/{limit} " f"({counter / limit * 100:.2f}%)"
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
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "google/gemma-2-27b-it",
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
        "--category",
        type=str,
        default="objects",
        help="Category Type",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="select_one",
        help="Task Type",
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
        default="selection/cached_states",
        help="Directory to save cached states",
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

    # # fusing the trained deltas
    # SYNTH_DATASET = "64"
    # checkpoint_path = os.path.join(
    #     env_utils.DEFAULT_RESULTS_DIR,
    #     "trained_params",
    #     f"{SYNTH_DATASET}",
    #     "_full__clamp=0.001",
    #     args.model.split("/")[-1],
    # )
    # version = "epoch_1"
    # checkpoint_path = os.path.join(
    #     env_utils.DEFAULT_RESULTS_DIR, checkpoint_path, version
    # )
    # checkpoint_path = os.path.join(checkpoint_path, "trainable_params.pt")
    # loaded_deltas = torch.load(checkpoint_path, map_location="cpu")
    # free_gpu_cache()
    # TrainableLM_delta.fuse_with_model(mt._model, loaded_deltas)

    # locations
    all_layers = [mt.embedder_name]  # embeddings
    for layer_idx in range(mt.n_layer):
        all_layers.extend(
            [
                mt.attn_module_name_format.format(layer_idx),  # attn
                mt.mlp_module_name_format.format(layer_idx),  # mlp
                mt.layer_name_format.format(layer_idx),  # residual
            ]
        )

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
        args.token_pos,
    )
    os.makedirs(save_dir, exist_ok=True)

    cache_selection_states(
        mt=mt,
        category_type=args.category,
        task_name=args.task,
        limit=args.limit,
        token_position=args.token_pos,
        modules=all_layers,
        save_dir=save_dir,
    )

import argparse
import json
import logging
import os
from datasets import load_dataset
import torch
from tqdm import tqdm

from src.functional import generate_with_patch, free_gpu_cache
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
from typing import Literal

logger = logging.getLogger(__name__)

def load_gsm8k_data(data_path: str = "trimmed_r1_responses.json") -> list[str]:
    """Load GSM8k dataset from the specified path."""
    full_path = os.path.join(env_utils.DEFAULT_DATA_DIR, data_path)
    logger.info(f"Loading GSM8k data from {full_path}")
    
    with open(full_path, "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} questions from GSM8k dataset")
    return [d["question"] for d in data]

def load_reasoning_data(ds_name: Literal["SoAp9035/r1-reasoning-tr", "KingNish/mini_reasoning_1k"]) -> list[str]:
    reasoning_ds = load_dataset(ds_name)
    key = "instructions" if ds_name == "SoAp9035/r1-reasoning-tr" else "question"
    return [d[key] for d in reasoning_ds["train"]]

def cache_thinking_responses(
    model_key: str,
    reasoning_data: list[str],
    max_samples: int = -1,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    save_dir: str = "cached_thinking",
    file_name: str = "reasoning.json"
):
    """Cache model responses with thinking tokens for GSM8k questions."""
    logger.info(f"Initializing model: {model_key}")
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.bfloat16,
    )
    
    # Create output directory
    model_name = model_key.split('/')[-1]
    output_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, save_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)
    
    # Limit samples if specified
    if max_samples > 0:
        reasoning_data = reasoning_data[:max_samples]
    
    results = []
    
    # Process questions
    for i, question in enumerate(tqdm(reasoning_data, desc=f"Processing with {model_name}")):
        
        # For 50% of questions, just append <think> token
        # For the other 50%, use the chat template with enable_thinking=True
        if i % 2 == 0 and False: # don't use this option for now
            # Direct append <think> token
            prompt = question + "<think>"
            logger.debug(f"Direct token prompt: {prompt[:50]}...")
        else:
            # Use chat template
            messages = [{"role": "user", "content": question}]
            prompt = mt.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            logger.debug(f"Chat template prompt: {prompt[:50]}...")
        
        # Generate response
        try:
            response = generate_with_patch(
                mt=mt,
                inputs=prompt,
                n_gen_per_prompt=1,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )[0]
            
            # Store result
            results.append({
                "question": question,
                "prompt": prompt,
                "response": response
            })
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(reasoning_data)} questions")
                # Save intermediate results
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
        
        # Clean up memory
        if (i + 1) % 5 == 0:
            free_gpu_cache()
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved {len(results)} responses to {output_file}")
    return results


if __name__ == "__main__":
    ##################################################################################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ##################################################################################################
    parser = argparse.ArgumentParser(description="Cache thinking process responses from LLMs on GSM8k questions")
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model identifier from HuggingFace or local path",
    )
    
    parser.add_argument(
        "--gsm8k",
        type=str,
        default="trimmed_r1_responses.json",
        help="Filename of GSM8k data in the data directory",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)",
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation",
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="cached_thinking",
        help="Directory name within results folder to save responses",
    )

    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        choices=["SoAp9035/r1-reasoning-tr", "KingNish/mini_reasoning_1k"],
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")
    
    if args.hf_dataset is None:
        # Load GSM8k data
        reasoning_ds = load_gsm8k_data(args.gsm8k)
        filename = "gsm8k.json"
    else:
        # Load reasoning data from HuggingFace dataset
        reasoning_ds = load_reasoning_data(args.hf_dataset)
        filename = f"{args.hf_dataset.split('/')[-1]}.json"
    
    # Cache thinking responses
    cache_thinking_responses(
        model_key=args.model,
        reasoning_data=reasoning_ds,
        max_samples=args.max_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        save_dir=args.save_dir,
        file_name=filename,
    )
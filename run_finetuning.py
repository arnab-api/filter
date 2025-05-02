import os

MODELS = [
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.1-8B",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    "Qwen/Qwen3-1.7B"
]
TRAIN_DOC = "synthetic_entities_bio.json"
REG_LIMIT = 10000
BATCH_SIZE = 4
MAX_EPOCHS = 30
SAVE_PATH = "trained_params"
LORA_RANKS = [None, 256]

cmd_template = 'python -m scripts.train --model="{}" -v'

for model in MODELS:
    for lora in LORA_RANKS:
        print("#" * 80)
        print(f"Finetuning Model: {model} | {lora=}")

        cmd = cmd_template.format(model)
        cmd += f" --max_epochs={MAX_EPOCHS}"
        cmd += f" --batch_size={BATCH_SIZE}"
        cmd += f" --reg_limit={REG_LIMIT}"
        cmd += f" --train_doc={TRAIN_DOC}"
        cur_run_name = f"{model.split('/')[-1]}_BIO"
        cur_save_path = SAVE_PATH
        if lora is not None:
            cur_run_name += f"_lora_{lora}"
            cur_save_path += os.path.join(f"lora_{lora}")
        else:
            cur_run_name += "_full"
            cur_save_path += os.path.join("full")

        cmd += f' --run_name="{cur_run_name}"'
        cmd += f' --save_path="{cur_save_path}"'
        if lora is not None:
            cmd += f" --lora_rank={lora}"

        logs_dir = f"logs/{model.split('/')[-1]}"
        os.makedirs(logs_dir, exist_ok=True)

        log_file_name = "full" if lora is None else f"lora_{lora}"
        cmd += f" 2>&1 | tee {logs_dir}/{log_file_name}.log"

        print(cmd)
        print("#" * 80)

        os.system(cmd)

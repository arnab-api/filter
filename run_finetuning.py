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
SAVE_PATH = "delta_weights_bio"

cmd_template = (
    'python -m scripts.full_finetune --max_epochs=100 --batch_size=8 --model="{}" -v'
)

for model in MODELS:
    print("#" * 80)
    print(f"Finetuning Model: {model}")
    print("#" * 80)

    cmd = cmd_template.format(model)
    cmd += f" --train_doc={TRAIN_DOC}"
    cmd += f" --batch_size={BATCH_SIZE}"
    cmd += f" --reg_limit={REG_LIMIT}"
    cmd += f' --run_name="{model.split("/")[-1]}_BIO"'
    cmd += f' --save_path="{SAVE_PATH}"'

    logs_dir = f"logs/{model.split('/')[-1]}"
    os.makedirs(logs_dir, exist_ok=True)
    cmd += f" 2>&1 | tee {logs_dir}/ft_bio.log"
    print(cmd)
    os.system(cmd)

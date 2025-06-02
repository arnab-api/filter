import os

MODELS = [
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.1-8B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    "meta-llama/Llama-3.3-70B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct",
]
# SYNTH_DATASET = "icosahedron_1"
SYNTH_DATASET = "test_72"

TRAIN_DOC_PATH = f"synthetic_entities/{SYNTH_DATASET}"
SAVE_PATH = f"trained_params/{SYNTH_DATASET}"
REG_LIMIT = 15000
BATCH_SIZE = 8
MAX_EPOCHS = 10
SAVE_INTERVAL = 5
WARMUP_STEPS = 1000

# LORA_RANKS = [None, 512]
LORA_RANKS = [None]
CLAMP_ABS_VALUE = 1e-3
UPTO_LAYER = 60  # None means full model
LAYER_STEP = 1

cmd_template = 'python -m scripts.train --model="{}" -v'

for model in MODELS:
    for lora in LORA_RANKS:
        print("#" * 80)
        print(f"Finetuning Model: {model} | {lora=}")

        cmd = cmd_template.format(model)
        cmd += f" --max_epochs={MAX_EPOCHS}"
        cmd += f" --batch_size={BATCH_SIZE}"
        cmd += f" --reg_limit={REG_LIMIT}"
        cmd += f" --train_doc={TRAIN_DOC_PATH}"
        cmd += f" --save_interval={SAVE_INTERVAL}"
        cmd += f" --warmup_steps={WARMUP_STEPS}"

        cur_run_name = f"{model.split('/')[-1]}"
        cur_save_path = SAVE_PATH
        if lora is not None:
            cur_run_name += f"_lora_{lora}"
            cur_save_path = os.path.join(cur_save_path, f"_lora_{lora}")
        else:
            cur_run_name += f"_full__clamp={CLAMP_ABS_VALUE}"
            cur_save_path = os.path.join(
                cur_save_path, f"_full__clamp={CLAMP_ABS_VALUE}"
            )
            if CLAMP_ABS_VALUE is not None:
                cmd += f" --clamp_abs_value={CLAMP_ABS_VALUE}"

        cur_run_name += f"_{SYNTH_DATASET}"

        cmd += f' --run_name="{cur_run_name}"'
        # cmd += " --run_name='Test'"
        cmd += f' --save_path="{cur_save_path}"'
        if lora is not None:
            cmd += f" --lora_rank={lora}"

        if UPTO_LAYER is not None:
            cmd += f" --upto_layer={UPTO_LAYER}"

        if LAYER_STEP != 1:
            cmd += f" --layer_step={LAYER_STEP}"

        cmd += " --skip_thinking_reg"
        cmd += " --use_8bit"

        logs_dir = f"logs/{SYNTH_DATASET}/{model.split('/')[-1]}"
        os.makedirs(logs_dir, exist_ok=True)

        log_file_name = (
            f"full__clamp={CLAMP_ABS_VALUE}" if lora is None else f"lora_{lora}"
        )
        log_file_name += f"_{SYNTH_DATASET}"
        cmd += f" 2>&1 | tee {logs_dir}/{log_file_name}.log"

        print(cmd)
        print("#" * 80)

        os.system(cmd)

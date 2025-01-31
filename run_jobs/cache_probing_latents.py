import os
import time

import numpy as np

######################################################
np.random.seed(123456)
######################################################


PROBE_CLASSES = [
    # "atheletes/baseball",
    # "atheletes/basketball",
    # "atheletes/cricket",
    # "atheletes/golf",
    # "atheletes/soccer",
    # "atheletes/tennis",
    # "profession/actors",
    "profession/chefs",
    "profession/musicians",
    "profession/politicians",
    "profession/scientists",
    "profession/writers",
]

MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.2-3B": "llama-3b",
    "meta-llama/Llama-3.1-8B": "llama-8b",
}

# CUDA_OPTIONS = list(range(8))
CUDA_OPTIONS = [1, 2, 3, 4, 5, 6, 7]
LIMIT = 10000
SAVE_DIR = "probing_latents"

job_path = "job_scripts/" + str(time.ctime()).replace(" ", "_")
print("##################################################################")
print(job_path)
print("------------------------------------------------------------------")
os.makedirs(job_path, exist_ok=True)

exp_name_2_cmd: dict = {}

assert len(CUDA_OPTIONS) >= len(PROBE_CLASSES)
for idx, probe_class in enumerate(PROBE_CLASSES):
    cuda_idx = CUDA_OPTIONS[idx]
    for model in MODELS:
        model_short_name = MODEL_SHORT_NAMES[model]
        cmd = "python -m scripts.cache_probing_latents"
        cmd += f' --model="{model}"'
        cmd += f' --probe_class="{probe_class}"'
        cmd += f" --limit={LIMIT}"
        cmd += f' --save_dir="{SAVE_DIR}"'
        cmd += f" --seed=123456"
        cmd += f" |& tee logs/{model_short_name}/{probe_class.split('/')[-1]}.log"

        exp_name = f"{model_short_name}__{probe_class.split('/')[-1]}"
        exp_name_2_cmd[exp_name] = f"CUDA_VISIBLE_DEVICES={cuda_idx} {cmd}"


### saving the jobs
for exp_name in exp_name_2_cmd:
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(exp_name_2_cmd[exp_name])

    with open(f"{job_path}/{exp_name}.sh", "w") as f:
        f.writelines(bash_template)

## running the jobs
for job in os.listdir(job_path):
    job_script = f"{job_path}/{job}"
    exp_name = job.split(".")[0]
    cmd = f"screen -dmS {exp_name} bash {job_script}"
    print("submitting job: ", job)
    print(cmd)
    os.system(cmd)
    print("\n")

print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
print("##################################################################")

import itertools
import os
import time
from dataclasses import dataclass

import numpy as np
import yaml

######################################################
np.random.seed(123456)
######################################################


class MachineConfig:
    name: str
    cuda_options: list[int]

    def __init__(self, name: str, cuda_options: list[int] = [0]):
        self.name = name
        self.cuda_options = cuda_options


MACHINES = [
    MachineConfig(name="local"),
    # MachineConfig(name="umibozu"),
    # MachineConfig(name="saitama", cuda_options=[0, 1]),
    # MachineConfig(
    #     name="nagoya.research.khoury.northeastern.edu", cuda_options=list(range(8))
    # ),
]


WORKERS = []
for machine in MACHINES:
    for cuda_option in machine.cuda_options:
        WORKERS.append((machine.name, cuda_option))

PROJECT_ROOT = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])
with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
    config = yaml.safe_load(f)
    username = config["BAULAB_USER"]
    password = config["BAULAB_PASS"]

print(f"WORKERS ({len(WORKERS)}): {WORKERS}")

# --------------------------------------------------------------------------------

PROBE_CLASSES = [
    # "atheletes/baseball",
    # "atheletes/basketball",
    # "atheletes/cricket",
    # "atheletes/golf",
    # "atheletes/soccer",
    # "atheletes/tennis",
    # "profession/actors",
    "profession/chefs",
    # "profession/musicians",
    # "profession/politicians",
    # "profession/scientists",
    # "profession/writers",
]

MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.2-3B": "llama-3b",
    "meta-llama/Llama-3.1-8B": "llama-8b",
}


LIMIT = 10000
SAVE_DIR = "probing_latents"

# --------------------------------------------------------------------------------

job_path = "job_scripts/" + str(time.ctime()).replace(" ", "_")
print("##################################################################")
print(job_path)
print("------------------------------------------------------------------")
os.makedirs(job_path, exist_ok=True)


assert len(WORKERS) >= len(PROBE_CLASSES)

exp_name_2_cmd: dict = {}
for idx, probe_class in enumerate(PROBE_CLASSES):
    machine, cuda_idx = WORKERS[idx]
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

        exp_name_2_cmd[exp_name] = {
            "machine": machine,
            "cmd": f"CUDA_VISIBLE_DEVICES={cuda_idx} {cmd}",
        }

for exp_name in exp_name_2_cmd:
    machine = exp_name_2_cmd[exp_name]["machine"]
    cmd = exp_name_2_cmd[exp_name]["cmd"]
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(f"\n# {exp_name=} | {machine=}\n")
        bash_template.append(f"{cmd}\n")

    ### save the job script locally
    with open(f"{job_path}/{exp_name}.sh", "w") as f:
        f.writelines(bash_template)

    ### copy the job script to the remote machine .remote_jobs with a new file name
    file_name = str(time.ctime()).replace(" ", "_") + "___" + exp_name + ".sh"
    file_path = os.path.join("~/Codes/Projects/retrieval/.remote_jobs/", file_name)

    # copy the job script to the remote machine and then run it
    if machine != "local":
        os.system(
            f"sshpass -p {password} scp {job_path}/{exp_name}.sh {username}@{machine}:{file_path}"
        )
        os.system(
            f"sshpass -p {password} ssh {username}@{machine} 'screen -dmS {exp_name} bash {file_path}'"
        )
    else:
        os.system(f"cp {job_path}/{exp_name}.sh {file_path}")
        os.system(f"screen -dmS {exp_name} bash {file_path}")

    print(f"submitted {exp_name} to {machine} with cmd: {cmd}")


print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
print("##################################################################")

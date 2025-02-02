import itertools
import os
import time
from dataclasses import dataclass

import numpy as np
import yaml

######################################################
np.random.seed(123456)
######################################################

PROJECT_ROOT = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])
with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
    config = yaml.safe_load(f)
    username = config["BAULAB_USER"]
    password = config["BAULAB_PASS"]


class MachineConfig:
    name: str
    cuda_options: list[int]

    def __init__(self, name: str, cuda_options: list[int] = [0]):
        self.name = name
        self.cuda_options = cuda_options


MACHINES = [
    MachineConfig(
        name="local",
        # cuda_options=[0],
        cuda_options=list(range(8)),
    ),
    # MachineConfig(name="umibozu"),
    # MachineConfig(name="tokyo"),
    # MachineConfig(name="osaka"),
    # MachineConfig(name="hamada"),
    # MachineConfig(name="karasuno"),
    # MachineConfig(name="kumamoto"),
    # MachineConfig(name="fukuyama"),
    # MachineConfig(name="sendai"),
    # MachineConfig(name="andromeda"),
    # MachineConfig(name="ei"),
    # MachineConfig(name="kobe"),
    # MachineConfig(name="macondo"),
    # MachineConfig(name="hokkaido"),
    # MachineConfig(name="karakuri"),
    # MachineConfig(name="bippu"),
    # MachineConfig(name="hawaii"),
    # MachineConfig(name="kyoto", cuda_options=[0, 1]),
    # MachineConfig(name="saitama", cuda_options=[0, 1]),
    # MachineConfig(
    #     name="nagoya.research.khoury.northeastern.edu", cuda_options=list(range(8))
    # ),
    # MachineConfig(
    #     name="hakone", cuda_options=list(range(8))
    # ),
]


WORKERS = []
for machine in MACHINES:
    for cuda_option in machine.cuda_options:
        WORKERS.append((machine.name, cuda_option))

print(f"WORKERS ({len(WORKERS)}): {WORKERS}")

# --------------------------------------------------------------------------------

MODELS = {
    "meta-llama/Llama-3.2-3B": {
        "n_layers": 24,
        "layer_name_format": "model.layers.{}",
    },
    "meta-llama/Llama-3.1-8B": {
        "n_layers": 32,
        "layer_name_format": "model.layers.{}",
    },
}


LIMIT = 6000
SAVE_DIR = "linear_probes"

MODEL_KEY = "meta-llama/Llama-3.2-3B"
TOKEN_IDX = -1

LAYERS = list(range(MODELS[MODEL_KEY]["n_layers"]))
LAYER_SPLITS = np.array_split(LAYERS, len(WORKERS))
LAYER_NAME_FORMAT = MODELS[MODEL_KEY]["layer_name_format"]

# --------------------------------------------------------------------------------

job_path = "job_scripts/" + str(time.ctime()).replace(" ", "_")
print("##################################################################")
print(job_path)
print("------------------------------------------------------------------")
os.makedirs(job_path, exist_ok=True)

exp_name_2_cmd: dict = {}
for idx, layers in enumerate(LAYER_SPLITS):
    machine, cuda_idx = WORKERS[idx]

    model_short_name = MODEL_KEY.split("/")[-1]
    cmd = "python -m scripts.train_probes"
    cmd += f' --model="{MODEL_KEY}"'
    cmd += f" --token_idx={TOKEN_IDX}"
    cmd += f' --layer_name_format="{LAYER_NAME_FORMAT}"'
    cmd += f' --layers {" ".join(map(str, layers))}'
    cmd += f" --limit={LIMIT}"
    cmd += f' --save_dir="{SAVE_DIR}"'

    cmd += f" --seed=123456"

    layer_short_hand = "_".join(map(str, layers))
    cmd += f" |& tee logs/train_probes/{model_short_name}/{layer_short_hand}.log"

    exp_name = f"{model_short_name}__{layer_short_hand}"

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
    project_path = "~/Codes/Projects/retrieval/"
    file_path = os.path.join(project_path, ".remote_jobs/", file_name)

    # copy the job script to the remote machine and then run it
    if machine != "local":
        os.system(
            f"sshpass -p {password} scp {job_path}/{exp_name}.sh {username}@{machine}:{file_path}"
        )
        os.system(
            f"sshpass -p {password} ssh {username}@{machine} 'screen -dmS {exp_name} bash -i {file_path}'"
        )
    else:
        os.system(f"cp {job_path}/{exp_name}.sh {file_path}")
        os.system(f"screen -dmS {exp_name} bash {file_path}")

    print(f"submitted {exp_name} to {machine} with script {file_path}")


print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
print("##################################################################")

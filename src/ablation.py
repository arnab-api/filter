import logging
import os
from typing import Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


def load_mean_states(cache_dir: Optional[str]):
    mean_states = {}
    total_tokens = 0
    for doc_cache in os.listdir(cache_dir):
        npz_file = np.load(os.path.join(cache_dir, doc_cache), allow_pickle=True)
        input_ids = npz_file["input_ids"]
        token_count = input_ids.shape[1]
        outputs = npz_file["outputs"].item()
        for key, value in outputs.items():
            if key not in mean_states:
                mean_states[key] = value * token_count
            else:
                mean_states[key] += value * token_count
        total_tokens += token_count

    for key in mean_states.keys():
        mean_states[key] = torch.Tensor(mean_states[key]).squeeze() / total_tokens

    return mean_states


########################## MLP ##########################

#########################################################


########################### ATTN ###########################


############################################################

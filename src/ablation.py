import copy
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm

from src.dataset import BridgeDataset, BridgeSample, Relation
from src.models import ModelandTokenizer, is_llama_variant
from src.tokens import find_token_range, insert_padding_before_pos, prepare_input
from src.utils.env_utils import CLAUDE_CACHE_DIR, GPT_4O_CACHE_DIR
from src.utils.typing import SVD, ArrayLike, PredictedToken, Tokenizer, TokenizerOutput

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

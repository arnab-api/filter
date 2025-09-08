import logging
from posix import preadv
from typing import Optional

import torch

from src.functional import get_hs, interpret_logits
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.tokens import prepare_input
from src.utils.typing import PredictedToken, TokenizerOutput

logger = logging.getLogger(__name__)


def get_first_token_id(name, tokenizer, prefix=" "):
    """Get the first token ID for a given name."""
    # print(f"{prefix=} | {name=}")
    return (
        tokenizer(prefix + name, return_tensors="pt", add_special_tokens=False)
        .input_ids[0][0]
        .item()
    )


class KeyedSet:
    def __init__(self, items, tokenizer):
        self.tokenizer = unwrap_tokenizer(tokenizer)
        self._dict = {get_first_token_id(item, tokenizer): item for item in items}

    def __sub__(self, other):
        diff_keys = set(self._dict.keys()) - set(other._dict.keys())
        values = [self._dict[k] for k in diff_keys]
        return KeyedSet(values, self.tokenizer)

    @property
    def keys(self):
        return list(self._dict.keys())

    @property
    def values(self):
        return list(self._dict.values())

    @property
    def len(self):
        return len(self._dict)

    def show(self):
        for k, v in self._dict.items():
            print(f'{k}["{self.tokenizer.decode(k)}"]: {v}')


# people_by_prof_set = {k: KeyedSet(v) for k, v in people_by_prof.items()}

# people_by_prof_set["actor"].show()
# print("-" * 50)
# people_by_prof_set["chef"].show()
# print("-" * 50)
# (people_by_prof_set["actor"] - people_by_prof_set["chef"]).show()


def verify_correct_option(
    mt: ModelandTokenizer,
    target: int | str,
    options: list[str] | list[int],
    input: Optional[str | TokenizerOutput] = None,
    logits: torch.Tensor | None = None,
    prefix: str = " ",
    is_counting_task: bool = False,
    **kwargs,
) -> tuple[bool, list[PredictedToken], dict[int, tuple[int, PredictedToken]]]:
    assert (
        logits is not None or input is not None
    ), "Either logits or input must be provided."
    if logits is None:
        input = (
            prepare_input(
                prompts=input,
                tokenizer=mt,
            )
            if isinstance(input, str)
            else input
        )
        logit_module = (mt.lm_head_name, -1)
        logits = get_hs(
            mt=mt, input=input, locations=[logit_module], return_dict=False
        ).squeeze()

    target = (
        get_first_token_id(target, mt.tokenizer, prefix=prefix)
        if isinstance(target, str)
        else target
    )
    options = [
        (
            get_first_token_id(opt, mt.tokenizer, prefix=prefix)
            if isinstance(opt, str)
            else opt
        )
        for opt in options
    ]
    if is_counting_task:
        predictions, track_options = interpret_logits(
            tokenizer=mt.tokenizer,
            logits=logits,
            interested_tokens=options,
            **kwargs,
        )
        correct = predictions[0].token_id == target
        print(f"{predictions[0].token_id=}")
        return correct, predictions, track_options
    else:
        predictions, track_options = interpret_logits(
            tokenizer=mt.tokenizer,
            logits=logits,
            interested_tokens=options,
            **kwargs,
        )
        option_scores = [pred for obj_tok, (obj_rank, pred) in track_options.items()]
        option_scores = sorted(option_scores, key=lambda x: x.logit, reverse=True)
        correct = option_scores[0].token_id == target
        return correct, predictions, track_options

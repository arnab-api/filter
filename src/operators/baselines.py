import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from src.functional import (filter_subspace_information, free_gpu_cache,
                            get_module_nnsight, interpret_logits)
from src.models import ModelandTokenizer
from src.operators.utils import project_to_vocab
from src.utils.typing import PredictedToken

logger = logging.getLogger(__name__)


from src.operators.operators import Operator, OperatorOutput
from src.operators.utils import project_to_vocab


@dataclass(frozen=False, kw_only=True)
class LogitLens(Operator):
    class_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.layer is not None:
            logger.warning(
                f"({type(self)}) layer={self.layer} will be ignored in __call__()."
            )

    @torch.inference_mode()
    def __call__(
        self,
        h: torch.Tensor,
        interested_indices: list[int] = [],
    ):

        top_pred, class_pred = project_to_vocab(
            mt=self.mt,
            h=h,
            layer_name=self.mt.layer_names[-1],  # set output of the last layer
            interested_tokens=list(set(self.class_indices + interested_indices)),
        )

        return OperatorOutput(
            top_predictions=top_pred,
            class_predictions=sorted(
                [pred for rank, pred in class_pred.values()],
                key=lambda x: x.prob,
                reverse=True,
            ),
        )

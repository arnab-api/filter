import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from src.functional import free_gpu_cache, get_module_nnsight, interpret_logits
from src.models import ModelandTokenizer
from src.operators.utils import project_to_vocab
from src.utils.typing import ArrayLike, PredictedToken

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class OperatorOutput:
    """Predicted object tokens and their probabilities under the decoder head."""

    top_predictions: list[PredictedToken]
    class_predictions: Optional[list[PredictedToken]] = None
    logits: Optional[ArrayLike] = None
    corner_h: Optional[torch.Tensor] = None


@dataclass(frozen=True, kw_only=True)
class Basis:
    direction: torch.Tensor
    z: str
    token_idx: list[int] = field(default_factory=list)
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=False, kw_only=True)
class Operator:
    """An abstract relation operator, which maps subjects to objects."""

    mt: ModelandTokenizer
    layer: Optional[str] = None

    def __call__(self, **kwargs: Any) -> OperatorOutput:
        raise NotImplementedError


@dataclass(frozen=False, kw_only=True)
class BasisOperator(Operator):
    # ! Doesn't work well
    """
    Equation 1 in the writeup
    """

    concept_directions: list[Basis]
    _projection_matrix: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        assert self.layer is not None, "Layer must be specified."

    @property
    def projection_matrix(self) -> torch.Tensor:
        if hasattr(self, "_projection_matrix"):
            return self._projection_matrix

        device = self.mt.device
        Q = torch.stack(
            [basis.direction for basis in self.concept_directions], dim=1
        ).to(device)
        Q = Q.to(device)
        Q = Q / Q.norm(dim=0)
        projection = Q @ (Q.T @ Q).pinverse() @ Q.T
        self._projection_matrix = projection

        return projection

    def __call__(self, h: torch.Tensor, project_to_subspace: bool = True) -> list[dict]:
        if project_to_subspace:
            h = h / h.norm()
            h = self.projection_matrix @ h

        similarities = [
            dict(
                sim=torch.nn.functional.cosine_similarity(h, basis.direction, dim=0),
                z=basis.z,
            )
            for basis in self.concept_directions
        ]

        return sorted(similarities, key=lambda x: x["sim"], reverse=True)


@dataclass(frozen=False, kw_only=True)
class CornerOperator(Operator):
    corner: torch.Tensor
    beta: float = 1.0
    class_indices: list[int] = field(default_factory=list)
    training_args: dict[str, Any] = field(default_factory=dict)
    corner_interpretation: list[PredictedToken] = field(default_factory=list)
    concept_subspace: torch.Tensor | None = None

    def __post_init__(self):
        assert self.layer is not None, "Layer must be specified."

    def __str__(self):
        return f"{self.interpret()}"

    def interpret(self):
        return project_to_vocab(mt=self.mt, h=self.corner, layer_name=self.layer)

    def __call__(
        self,
        h: torch.Tensor,
        beta: float | None = None,
        interested_indices: list[int] = [],
        return_logits: bool = False,
        return_corner_h: bool = False,
    ) -> OperatorOutput:
        beta = self.beta if beta is None else beta
        device = self.mt.device
        h = h.to(device)
        corner = self.corner.to(device)

        # normalize the corner to have the same magnitude as beta * |h|
        corner = beta * (corner / corner.norm()) * h.norm()

        # add corner to h and normalize
        h = (h + corner) / 2.0

        if self.concept_subspace is not None:
            logger.info(f"{h.norm()=}")

            # TODO: (debug) doesn't make sense why it's not working
            # # in principle, this should only be done at the last layer, maybe with SGD
            # h = (
            #     torch.eye(self.mt.n_embd, device=h.device) - self.zero_out_projection
            # ) @ h
            # h = self.concept_subspace @ h

            # tune h in current layer to project out from the last layer
            # h = filter_subspace_information(
            #     mt=self.mt,
            #     h=h,
            #     layer_name=self.mt.layer_names[-1],  #! not sure why this makes sense
            #     subspace=self.concept_subspace,
            # )

            # don't do anything
            h = h

            logger.info(f"{h.norm()=}")

        inputs = self.mt.tokenizer(
            self.mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        )
        with torch.inference_mode():
            with self.mt.trace(inputs):
                module = get_module_nnsight(self.mt, self.layer)
                module.output[0][0, :] = h
                logits = self.mt.output.logits[0, -1].save()

        # TODO: Also return which token is getting most promoted compared to its relative position in the corner rep itself
        top_pred, class_pred = interpret_logits(
            tokenizer=self.mt,
            logits=logits,
            interested_tokens=list(set(self.class_indices + interested_indices)),
        )

        free_gpu_cache()

        return OperatorOutput(
            top_predictions=top_pred,
            class_predictions=sorted(
                [pred for rank, pred in class_pred.values()],
                key=lambda x: x.prob,
                reverse=True,
            ),
            logits=logits if return_logits else None,
            corner_h=h if return_corner_h else None,
        )

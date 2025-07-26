import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike, PredictedToken, Tokenizer
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.tokens import prepare_input
from src.functional import detensorize, predict_next_token
from src.selection.utils import KeyedSet, get_first_token_id

logger = logging.getLogger(__name__)

@dataclass
class SelectionSample(DataClassJsonMixin):
    subj: str
    obj: str
    obj_idx: int
    options: Sequence[str]
    category: str
    prompt_template: str = """Which person from the following list has the {} in common with {}?
Options: {}.
Ans:"""
    prediction: Optional[Sequence[PredictedToken]] = None
    obj_token_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.options, Sequence):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")
        assert (
            self.options[self.obj_idx] == self.obj
        ), "Object must be one of the options and match the object index."

    def __str__(self):
        return f"{self.subj} -> {self.obj} ({self.obj_idx}): {self.options}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)
    
    @property
    def prompt(self) -> str:
        options_str = ", ".join(f"{opt}" for opt in self.options)
        return self.prompt_template.format(self.category, self.subj, options_str)

@dataclass
class ObjectwiseResult:
    rank: int
    pred: PredictedToken


@dataclass
class LayerwiseResult:
    predictions: list[PredictedToken]
    objs: dict[str, ObjectwiseResult]


@dataclass
class SelectionPatchingResult(DataClassJsonMixin):
    patch_sample: SelectionSample
    clean_sample: SelectionSample
    results: dict[str, LayerwiseResult]


@dataclass
class SelectionPatchingResult_Multi(DataClassJsonMixin):
    patch_sample_1: SelectionSample
    patch_sample_2: SelectionSample
    patch_prompt: str

    clean_sample_1: SelectionSample
    clean_sample_2: SelectionSample
    clean_prompt: str

    results: dict[str, LayerwiseResult]

def load_people_by_category(
    tokenizer: Tokenizer,
    path: PathLike = os.path.join(DEFAULT_DATA_DIR, "synthetic_entities/64/profiles.json"),
    category: str = "occupation"
):
    """Load people by profession from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    entity_by_category = defaultdict(list)
    for entity in data:
        entity_by_category[entity[category]].append(entity['name'])

    people_by_category = {
        k: KeyedSet(v, tokenizer=tokenizer) for k, v in entity_by_category.items()
    }
    
    logger.info(f"Loaded {len(people_by_category)} categories")
    return people_by_category

def get_random_sample(
    people_by_category: dict[str, KeyedSet],
    mt: ModelandTokenizer,
    category: str,
    attribute: str | None = None,
    subj: str | None = None,
    n_distractors: int = 5,
    filter_by_lm_prediction: bool = True,
    obj_idx: int | None = None,
    get_alt_obj: bool = False,  # TODO(arnab): Need to check accuracy with the alt obj as well
    exclude_objs: Sequence[str] = [],
    exclude_distractor_categories: Sequence[str] = [],
    insert_distractor: Sequence[tuple[str, int]] = [],
    retry_count: int = 0,
) -> SelectionSample:
    """
    Get a random sample with the specified attribute.
    """

    tokenizer = unwrap_tokenizer(mt)
    if not people_by_category:
        load_people_by_category(tokenizer)

    attribute = attribute or random.choice(list(people_by_category.keys()))
    if attribute not in people_by_category:
        raise ValueError(
            f"Attribute '{attribute}' not found in {people_by_category.keys()}."
        )
    kwargs = {
        "subj": subj,
        "obj_idx": obj_idx,
    }
    # print(f"Category: {category}")
    # print(people_by_category[category].values)
    subj = (
        random.choice(list(people_by_category[attribute].values))
        if subj is None
        else subj
    )
    # logger.debug(f"{subj=}")
    obj = random.choice(
        (
            people_by_category[attribute]
            - KeyedSet([subj] + exclude_objs, tokenizer=tokenizer)
        ).values
    )
    obj_token_id = get_first_token_id(obj, tokenizer, prefix=" ")
    if obj_idx is None:
        obj_idx = random.randint(0, n_distractors)
    # logger.debug(f"{obj=}, {obj_token_id=}, {obj_idx=}, {exclude_objs=}")

    obj_arr = [obj]
    if get_alt_obj:
        # Get an alternative object with the same attribute
        alt_obj = random.choice(
            (
                people_by_category[attribute]
                - KeyedSet([subj, obj] + exclude_objs, tokenizer=tokenizer)
            ).values
        )
        obj_arr.append(alt_obj)
    else:
        alt_obj = None

    distractors = []
    obj_set = KeyedSet(obj_arr + exclude_objs, tokenizer=tokenizer)
    other_attributes = random.sample(
        list(
            set(people_by_category.keys())
            - set([attribute] + exclude_distractor_categories)
        ),
        k=n_distractors,
    )
    # print(other_categories)
    for other_attribute in other_attributes:
        distractors.append(
            random.choice((people_by_category[other_attribute] - obj_set).values)
        )

    options = distractors[:obj_idx] + [obj] + distractors[obj_idx:]
    for dist, idx in insert_distractor:
        assert idx != obj_idx, "Cannot replace answer with a distractor."
        assert idx < len(options), "Distractor index out of range."
        options[idx] = dist
    # logger.debug(f"{options=}")

    metadata = {"attribute": attribute}
    if get_alt_obj:
        alt_obj_token_id = get_first_token_id(alt_obj, tokenizer, prefix=" ")
        metadata["alt_obj"] = (alt_obj, alt_obj_token_id)
    sample = SelectionSample(
        subj=subj,
        obj=obj,
        obj_idx=obj_idx,
        options=options,
        category=category,
        obj_token_id=obj_token_id,
        metadata=metadata,
    )

    if filter_by_lm_prediction:
        prompt = sample.prompt
        logger.info(f"\nPrompt: {prompt}")
        inputs = prepare_input(prompts=prompt, tokenizer=mt)
        sample.metadata["tokenized"] = inputs.data
        
        predictions = predict_next_token(
            mt=mt,
            inputs=inputs,
        )[0]
        if predictions[0].token_id != obj_token_id:
            logger.error(
                f"""Sample = {sample}
Top prediction {predictions[0]} does not match the object {obj}[{obj_token_id}, "{mt.tokenizer.decode(obj_token_id)}"].
Retry count: {retry_count + 1}. Retrying ...
"""
            )
            return get_random_sample(
                people_by_category=people_by_category,
                mt=mt,
                n_distractors=n_distractors,
                get_alt_obj=get_alt_obj,
                filter_by_lm_prediction=filter_by_lm_prediction,
                category=category,
                attribute=attribute,
                exclude_objs=exclude_objs,
                exclude_distractor_categories=exclude_distractor_categories,
                insert_distractor=insert_distractor,
                retry_count=retry_count + 1,
                **kwargs,
            )

        sample.prediction = predictions

    sample.metadata["retry_count"] = retry_count
    return sample
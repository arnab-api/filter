#! TODO(arnab):
# The sample classes should inherit from a common base class to avoid code duplication.

import copy
import json
import logging
import os
import random
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import torch
from dataclasses_json import DataClassJsonMixin

from src.functional import detensorize, predict_next_token
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.selection.utils import KeyedSet, get_first_token_id, verify_correct_option
from src.tokens import find_token_range, prepare_input
from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike, PredictedToken

logger = logging.getLogger(__name__)

index_to_order = {
    0: "first",
    1: "second",
    2: "third",
    3: "fourth",
    4: "fifth",
    5: "sixth",
    6: "seventh",
    7: "eighth",
    8: "ninth",
    9: "tenth",
}

COUNT_STR_MAP = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
}


########################################## <Sample Data Classes> ##########################################
@dataclass
class SelectionSample(DataClassJsonMixin):
    obj: str
    obj_idx: int
    prompt_template: str
    options: list[str]
    answer: str | None = None  # if obj != answer
    subj: str | None = None
    category: str | None = None
    prediction: Optional[list[PredictedToken]] = None
    ans_token_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line", "numbered", "bulleted"] = "single_line"
    option_label_start_from: str = "a"  # for numbered option style

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_pivot_entity_>" in self.prompt_template:
            assert self.subj is not None
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, list):
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

    def prompt(
        self, option_style: Literal["single_line", "numbered", "bulleted"] | None = None
    ) -> str:
        prompt = self.prompt_template
        if "<_pivot_entity_>" in prompt:
            prompt = prompt.replace("<_pivot_entity_>", self.subj)
        if "<_category_>" in prompt:
            prompt = prompt.replace("<_category_>", self.category)
        if "<_order_>" in prompt:
            prompt = prompt.replace(
                "<_order_>", index_to_order.get(self.obj_idx, str(self.obj_idx))
            )

        if option_style is None:
            option_style = self.default_option_style

        if option_style == "single_line":
            options_str = ", ".join(f"{opt}" for opt in self.options)
            options_str = f"Options: {options_str}."

        elif option_style == "numbered":
            options_str = "\n".join(
                f"{chr(ord(self.option_label_start_from) + i)}. {opt}"
                for i, opt in enumerate(self.options)
            )
        elif option_style == "bulleted":
            options_str = "\n".join(f"* {opt}" for opt in self.options)
        else:
            raise ValueError(f"Invalid option_style: {option_style}.")

        prompt = prompt.replace("<_options_>", options_str)

        return prompt


def MCQify_sample(
    tokenizer: ModelandTokenizer, sample: SelectionSample, start_from="a"
) -> SelectionSample:
    tokenizer = unwrap_tokenizer(tokenizer)
    sample = copy.deepcopy(sample)
    sample.default_option_style = "numbered"
    sample.option_label_start_from = start_from
    correct_option = chr(ord(start_from) + sample.obj_idx)
    sample.ans_token_id = get_first_token_id(
        name=correct_option, tokenizer=tokenizer, prefix=" "
    )
    sample.metadata["question_type"] = "MCQ"
    return sample


@dataclass
class SelectAllSample(DataClassJsonMixin):
    subj: str
    options: list[str]
    category: str
    prompt_template: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_pivot_entity_>" in self.prompt_template:
            assert self.subj is not None
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, Sequence):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")
        if len(self.options) > 11:
            raise ValueError("There must be less than eleven options.")

    def __str__(self):
        return f"{self.category}: {self.subj} -> {self.options}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)

    def prompt(self):
        prompt = self.prompt_template
        one_shot = prompt + " All of the above."

        if "<_pivot_entity_>" in prompt:
            prompt = prompt.replace("<_pivot_entity_>", self.subj)
            one_shot = one_shot.replace("<_pivot_entity_>", "Isaac Newton")

        if "<_category_>" in prompt:
            prompt = prompt.replace("<_category_>", self.category)
            one_shot = one_shot.replace("<_category_>", "profession")

        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        scientists = [
            "Albert Einstein",
            "Marie Curie",
            "Charles Darwin",
            "Galileo Galilei",
            "Stephen Hawking",
            "Nikola Tesla",
            "Thomas Edison",
            "Gregor Mendel",
            "Johannes Kepler",
            "Michael Faraday",
            "Niels Bohr",
        ]
        options_str = "\n".join(
            f"{letters[i]}) {opt}" for i, opt in enumerate(self.options)
        )
        one_shot_options_str = "\n".join(
            f"{letters[i]}) {sci}"
            for i, sci in enumerate(scientists[: len(self.options)])
        )

        prompt = prompt.replace("<_options_>", options_str)
        one_shot = one_shot.replace("<_options_>", one_shot_options_str)

        return one_shot + "\n\n" + prompt


@dataclass
class CountingSample(DataClassJsonMixin):
    prompt_template: str
    options: list[str]
    count: int
    category: str | None = None
    prediction: Optional[list[PredictedToken]] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line"] = "single_line"
    ans_token_id: int | None = None

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, list):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")

    def __str__(self):
        return f"{self.category} -> {self.options}: Ans: {self.count}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)

    def prompt(
        self, option_style: Literal["single_line", "numbered"] | None = None
    ) -> str:
        prompt = self.prompt_template
        if "<_category_>" in prompt:
            prompt = prompt.replace("<_category_>", self.category)

        if option_style is None:
            option_style = self.default_option_style

        if option_style == "single_line":
            options_str = ", ".join(f"{opt}" for opt in self.options)

        else:
            raise ValueError(f"Invalid option_style: {option_style}.")

        prompt = prompt.replace("<_options_>", options_str)

        return prompt


@dataclass
class YesNoSample(DataClassJsonMixin):
    prompt_template: str
    options: list[str]
    yes: bool
    category: str | None = None
    prediction: Optional[list[PredictedToken]] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line"] = "single_line"
    ans_token_id: int | None = None

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, list):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")

    def __str__(self):
        answer = "Yes" if self.yes else "No"
        return f"{self.category} -> {self.options}: Ans: {answer}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)

    def prompt(
        self, option_style: Literal["single_line", "numbered"] | None = None
    ) -> str:
        prompt = self.prompt_template
        if "<_category_>" in prompt:
            prompt = prompt.replace("<_category_>", self.category)

        if option_style is None:
            option_style = self.default_option_style

        if option_style == "single_line":
            options_str = ", ".join(f"{opt}" for opt in self.options)

        else:
            raise ValueError(f"Invalid option_style: {option_style}.")

        prompt = prompt.replace("<_options_>", options_str)

        return prompt


@dataclass
class DeductionSample(DataClassJsonMixin):
    prompt: str
    answer: str
    depth: int
    topic: str
    prediction: Optional[Sequence[PredictedToken]] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.prompt} -> {self.answer}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)


@dataclass
class CounterFactualSamplePair(DataClassJsonMixin):
    patch_sample: Union[SelectionSample, CountingSample, YesNoSample]
    clean_sample: Union[SelectionSample, CountingSample, YesNoSample]

    @staticmethod
    def sample_type_to_class():
        return {
            "selection": SelectionSample,
            "counting": CountingSample,
            "yes_no": YesNoSample,
        }

    def detensorize(self):
        for sample in [self.patch_sample, self.clean_sample]:
            class_name = type(sample).__name__
            type_to_name = {
                "SelectionSample": "selection",
                "CountingSample": "counting",
                "YesNoSample": "yes_no",
            }
            sample.metadata["sample_type"] = type_to_name[class_name]
        self.patch_sample.detensorize()
        self.clean_sample.detensorize()

    @staticmethod
    def from_dict(d):
        sample_type = d["patch_sample"]["metadata"].pop("sample_type")
        sample_cls = CounterFactualSamplePair.sample_type_to_class()[sample_type]
        patch_sample = sample_cls.from_dict(d["patch_sample"])
        sample_type = d["clean_sample"]["metadata"].pop("sample_type")
        sample_cls = CounterFactualSamplePair.sample_type_to_class()[sample_type]
        clean_sample = sample_cls.from_dict(d["clean_sample"])
        return CounterFactualSamplePair(
            patch_sample=patch_sample,
            clean_sample=clean_sample,
        )


########################################### </Sample Data Classes> ##########################################


########################################### <Result Data Classes> ##########################################
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


################################################### </Result Data Classes> ##########################################


########################################### <Task Data Classes> ##########################################
@dataclass
class SelectOneTask(DataClassJsonMixin):
    category_type: str
    prompt_templates: list[str]
    category_wise_examples: dict[str, list] = field(default_factory=dict)
    task_name: str = "select_one"
    exclude_categories: dict[str, list] = field(default_factory=dict)

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            print(list(data.keys()))
            return SelectOneTask(
                task_name="select_one",
                category_type=data.get("name", category_type),
                prompt_templates=data["prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
                exclude_categories=data.get("exclude_categories", {}),
            )

    def filter_single_token(self, tokenizer, prefix=" "):
        """
        Filter the examples to only include those with a single token.
        """
        filtered = {}
        for cat, examples in self.category_wise_examples.items():
            filtered[cat] = []
            for example in examples:
                tokenized = tokenizer(
                    prefix + example, return_tensors="pt", add_special_tokens=False
                )
                if len(tokenized["input_ids"][0]) == 1:
                    filtered[cat].append(example)
        self.category_wise_examples = filtered

    def exclude_for_category(self, category):
        if category in self.exclude_categories:
            return self.exclude_categories[category]
        else:
            return []

    @property
    def categories(self):
        """
        Returns the list of categories in the task.
        """
        return list(self.category_wise_examples.keys())

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        subj: str | None = None,
        n_distractors: int = 5,
        filter_by_lm_prediction: bool = False,
        obj_idx: int | None = None,
        get_alt_obj: bool = False,  # TODO(arnab): Need to check accuracy with the alt obj as well
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
        output_formatting: Literal["zero_shot", "object", "lettered"] = "zero_shot",
    ) -> SelectionSample:
        """
        Get a random sample with the specified attribute.
        """

        def format_answer(obj_idx, obj, formatting):
            if formatting == "lettered":
                answer = f"{chr(ord('a') + obj_idx)}. {obj}"
            elif formatting == "object":
                answer = obj
            return answer

        kwargs = {
            "subj": subj,
            "obj_idx": obj_idx,
            "prompt_template_idx": prompt_template_idx,
            "option_style": option_style,
            "output_formatting": output_formatting,
        }

        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)
        # print(f"Category: {category}")
        # print(people_by_category[category].values)

        category = category or random.choice(list(category_wise_examples.keys()))
        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )

        exclude_distractor_categories = list(
            set(exclude_distractor_categories + self.exclude_for_category(category))
        )

        subj = (
            random.choice(list(category_wise_examples[category].values))
            if subj is None
            else subj
        )

        # logger.debug(f"{subj=}")
        print(category, ">>", category_wise_examples[category].values)
        obj = random.choice(
            (
                category_wise_examples[category]
                - KeyedSet([subj] + exclude_objs, tokenizer=tokenizer)
            ).values
        )

        prefix = ""
        if output_formatting != "zero_shot":
            one_shot = self.get_random_sample(
                mt=mt,
                prompt_template_idx=prompt_template_idx,
                category=None,
                subj=None,
                n_distractors=n_distractors,
                filter_by_lm_prediction=False,
                obj_idx=None,
                get_alt_obj=False,
                exclude_objs=[obj],
                exclude_distractor_categories=[category],
                output_formatting="zero_shot",
                option_style=option_style,
            )
            one_shot_answer = format_answer(
                obj_idx=one_shot.obj_idx,
                obj=one_shot.obj,
                formatting=output_formatting,
            )
            prefix = f"{one_shot.prompt()} {one_shot_answer}\n\n"

        obj_token_id = get_first_token_id(obj, tokenizer, prefix=" ")
        if obj_idx is None:
            obj_idx = random.randint(0, n_distractors)
        # logger.debug(f"{obj=}, {obj_token_id=}, {obj_idx=}, {exclude_objs=}")

        obj_arr = [obj]
        if get_alt_obj:
            # Get an alternative object with the same attribute
            alt_obj = random.choice(
                (
                    category_wise_examples[category]
                    - KeyedSet([subj, obj] + exclude_objs, tokenizer=tokenizer)
                ).values
            )
            obj_arr.append(alt_obj)
        else:
            alt_obj = None

        distractors = []
        obj_set = KeyedSet(obj_arr + exclude_objs, tokenizer=tokenizer)
        # other_categories = random.sample(
        #     list(
        #         set(category_wise_examples.keys())
        #         - set([category] + exclude_distractor_categories)
        #     ),
        #     k=n_distractors,
        # )
        other_categories = random.choices(
            list(
                set(category_wise_examples.keys())
                - set([category] + exclude_distractor_categories)
            ),
            k=n_distractors,
        )
        # print(other_categories)
        for other_attribute in other_categories:
            distractors.append(
                random.choice(
                    (category_wise_examples[other_attribute] - obj_set).values
                )
            )

        options = distractors[:obj_idx] + [obj] + distractors[obj_idx:]
        for dist, idx in insert_distractor:
            assert idx != obj_idx, "Cannot replace answer with a distractor."
            assert idx < len(options), "Distractor index out of range."
            options[idx] = dist
        # logger.debug(f"{options=}")

        answer = obj
        if output_formatting != "zero_shot":
            answer = format_answer(
                obj_idx=obj_idx,
                obj=obj,
                formatting=output_formatting,
            )

        metadata = {}
        if get_alt_obj:
            alt_obj_token_id = get_first_token_id(
                alt_obj, tokenizer, prefix=" "
            )  #! alt_obj is not consistent with the formatting yet
            metadata["alt_obj"] = (alt_obj, alt_obj_token_id)
        sample = SelectionSample(
            subj=subj,
            obj=obj,
            obj_idx=obj_idx,
            options=options,
            category=category,
            ans_token_id=get_first_token_id(answer, tokenizer, prefix=" "),
            answer=answer,
            metadata=metadata,
            prompt_template=self.prompt_templates[prompt_template_idx],
            default_option_style=option_style,
        )

        sample.prompt_template = prefix + sample.prompt_template
        if "qwen" in mt.name.lower():
            sample.prompt_template = "# " + sample.prompt_template  # for attention sink

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=obj_token_id,
                options=options,
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
    Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {obj}[{obj_token_id}, "{mt.tokenizer.decode(obj_token_id)}"].
    Retry count: {retry_count + 1}. Retrying ...
    """
                )
                return self.get_random_sample(
                    mt=mt,
                    n_distractors=n_distractors,
                    get_alt_obj=get_alt_obj,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    category=category,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    **kwargs,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample

    def __str__(self):
        return f"""SelectOneTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""


@dataclass
class SelectOddOneOutTask(SelectOneTask):
    task_name: str = "select_odd_one_out"

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/profession.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return SelectOddOneOutTask(
                task_name="select_odd_one_out",
                category_type=data.get("name", category_type),
                prompt_templates=data["odd_one_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        distractor_category: str | None = None,
        obj_category: str | None = None,
        subj: str | None = None,
        n_distractors: int = 5,
        filter_by_lm_prediction: bool = False,
        obj_idx: int | None = None,
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
        output_formatting: Literal["zero_shot", "object", "lettered"] = "zero_shot",
        **kwargs: dict[str, Any],
    ) -> SelectionSample:
        """
        Get a random sample with the specified attribute.
        """

        if len(kwargs) > 0:
            logger.warning(
                f"{type(self)} >> Unused keyword arguments: {kwargs}. Please check the function signature."
            )

        def format_answer(obj_idx, obj, formatting):
            if formatting == "lettered":
                answer = f"{chr(ord('a') + obj_idx)}. {obj}"
            elif formatting == "object":
                answer = obj
            return answer

        kwargs = {
            "subj": subj,
            "obj_idx": obj_idx,
            "prompt_template_idx": prompt_template_idx,
            "option_style": option_style,
            "output_formatting": output_formatting,
        }

        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)
        # print(f"Category: {category}")
        # print(people_by_category[category].values)
        obj_category = obj_category or random.choice(
            list(category_wise_examples.keys())
        )

        distractor_category = distractor_category or random.choice(
            list(
                set(category_wise_examples.keys())
                - set(exclude_distractor_categories + [obj_category])
            )
        )
        if distractor_category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{distractor_category}' not found in {category_wise_examples.keys()}."
            )
        subj = (
            random.choice(list(category_wise_examples[distractor_category].values))
            if subj is None
            else subj
        )

        distractors = random.sample(
            (
                category_wise_examples[distractor_category]
                - KeyedSet(items=[subj] + exclude_objs, tokenizer=tokenizer)
            ).values,
            k=n_distractors,
        )

        # logger.debug(f"{subj=}")
        obj = random.choice(
            (
                category_wise_examples[obj_category]
                - KeyedSet([subj] + exclude_objs + distractors, tokenizer=tokenizer)
            ).values
        )
        obj_token_id = get_first_token_id(obj, tokenizer, prefix=" ")
        if obj_idx is None:
            obj_idx = random.randint(0, n_distractors)
        # logger.debug(f"{obj=}, {obj_token_id=}, {obj_idx=}, {exclude_objs=}")

        options = distractors[:obj_idx] + [obj] + distractors[obj_idx:]
        for dist, idx in insert_distractor:
            assert idx != obj_idx, "Cannot replace answer with a distractor."
            assert idx < len(options), "Distractor index out of range."
            options[idx] = dist
        # logger.debug(f"{options=}")

        prefix = ""
        if output_formatting != "zero_shot":
            one_shot = self.get_random_sample(
                mt=mt,
                prompt_template_idx=prompt_template_idx,
                distractor_category=None,
                subj=None,
                n_distractors=n_distractors,
                filter_by_lm_prediction=False,
                obj_idx=None,
                get_alt_obj=False,
                exclude_objs=[obj],
                exclude_distractor_categories=[distractor_category],
                output_formatting="zero_shot",
                option_style=option_style,
            )
            one_shot_answer = format_answer(
                obj_idx=one_shot.obj_idx,
                obj=one_shot.obj,
                formatting=output_formatting,
            )
            prefix = f"{one_shot.prompt()} {one_shot_answer}\n\n"

        answer = obj
        if output_formatting != "zero_shot":
            answer = format_answer(
                obj_idx=obj_idx,
                obj=obj,
                formatting=output_formatting,
            )

        metadata = {}
        sample = SelectionSample(
            subj=subj,
            obj=obj,
            obj_idx=obj_idx,
            options=options,
            category=distractor_category,
            ans_token_id=get_first_token_id(answer, tokenizer, prefix=" "),
            answer=answer,
            metadata=metadata,
            prompt_template=self.prompt_templates[prompt_template_idx],
            default_option_style=option_style,
        )
        sample.metadata["odd_category"] = obj_category

        sample.prompt_template = prefix + sample.prompt_template

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=obj_token_id,
                options=options,
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
    Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {obj}[{obj_token_id}, "{mt.tokenizer.decode(obj_token_id)}"].
    Retry count: {retry_count + 1}. Retrying ...
    """
                )
                return self.get_random_sample(
                    mt=mt,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    distractor_category=distractor_category,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    **kwargs,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample

    def __str__(self):
        return f"""SelectOddOneOutTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""


@dataclass
class SelectOrderTask(SelectOneTask):
    task_name: str = "select_order"

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return SelectOrderTask(
                task_name="select_order",
                category_type=data.get("name", category_type),
                prompt_templates=data["order_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 1,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        n_distractors: int = 5,
        filter_by_lm_prediction: bool = False,
        obj_idx: int | None = None,
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
        output_formatting: Literal["zero_shot", "object", "lettered"] = "zero_shot",
        **kwargs: dict[str, Any],
    ) -> SelectionSample:
        """
        Get a random sample with the specified attribute.
        """

        if len(kwargs) > 0:
            logger.warning(
                f"{type(self)} >> Unused keyword arguments: {kwargs}. Please check the function signature."
            )

        def format_answer(obj_idx, obj, formatting):
            if formatting == "lettered":
                answer = f"{chr(ord('a') + obj_idx)}. {obj}"
            elif formatting == "object":
                answer = obj
            return answer

        kwargs = {
            "obj_idx": obj_idx,
            "prompt_template_idx": prompt_template_idx,
            "option_style": option_style,
            "output_formatting": output_formatting,
        }

        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)

        category = category or random.choice(list(category_wise_examples.keys()))

        obj = random.choice(
            (
                category_wise_examples[category]
                - KeyedSet(exclude_objs, tokenizer=tokenizer)
            ).values
        )
        obj_arr = [obj]
        obj_token_id = get_first_token_id(obj, tokenizer, prefix=" ")
        if obj_idx is None:
            obj_idx = random.randint(0, n_distractors)
        # logger.debug(f"{obj=}, {obj_token_id=}, {obj_idx=}, {exclude_objs=}")

        distractors = []
        obj_set = KeyedSet(obj_arr + exclude_objs, tokenizer=tokenizer)
        other_categories = random.sample(
            list(
                set(category_wise_examples.keys())
                - set([category] + exclude_distractor_categories)
            ),
            k=n_distractors,
        )
        # print(other_categories)
        for other_attribute in other_categories:
            distractors.append(
                random.choice(
                    (category_wise_examples[other_attribute] - obj_set).values
                )
            )

        options = distractors[:obj_idx] + [obj] + distractors[obj_idx:]
        for dist, idx in insert_distractor:
            assert idx != obj_idx, "Cannot replace answer with a distractor."
            assert idx < len(options), "Distractor index out of range."
            options[idx] = dist
        # logger.debug(f"{options=}")

        prefix = ""
        if output_formatting != "zero_shot":
            one_shot = self.get_random_sample(
                mt=mt,
                prompt_template_idx=prompt_template_idx,
                category=None,
                n_distractors=n_distractors,
                filter_by_lm_prediction=False,
                obj_idx=None,
                exclude_objs=[obj],
                exclude_distractor_categories=[category],
                output_formatting="zero_shot",
                option_style=option_style,
            )
            one_shot_answer = format_answer(
                obj_idx=one_shot.obj_idx,
                obj=one_shot.obj,
                formatting=output_formatting,
            )
            prefix = f"{one_shot.prompt()} {one_shot_answer}\n\n"

        answer = obj
        if output_formatting != "zero_shot":
            answer = format_answer(
                obj_idx=obj_idx,
                obj=obj,
                formatting=output_formatting,
            )

        metadata = {}
        sample = SelectionSample(
            subj=None,
            obj=obj,
            obj_idx=obj_idx,
            options=options,
            category=category,
            ans_token_id=get_first_token_id(answer, tokenizer, prefix=" "),
            answer=answer,
            metadata=metadata,
            prompt_template=self.prompt_templates[prompt_template_idx],
            default_option_style=option_style,
        )

        sample.prompt_template = prefix + sample.prompt_template

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=obj_token_id,
                options=options,
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
    Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {obj}[{obj_token_id}, "{mt.tokenizer.decode(obj_token_id)}"].
    Retry count: {retry_count + 1}. Retrying ...
    """
                )
                return self.get_random_sample(
                    mt=mt,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    category=category,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    **kwargs,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample

    def __str__(self):
        return f"""SelectOrderTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""


@dataclass
class SelectAllTask(DataClassJsonMixin):
    category_type: str
    prompt_templates: list[str]
    category_wise_examples: dict[str, list] = field(default_factory=dict)
    task_name: str = "select_all"

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/profession.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return SelectAllTask(
                task_name="select_all",
                category_type=data.get("name", category_type),
                prompt_templates=data["prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        category: str | None = None,
        subj: str | None = None,
        n_options: int = 6,
        filter_by_lm_prediction: bool = True,
        retry_count: int = 0,
    ) -> SelectAllSample:
        """
        Get a random sample with the specified attribute.
        """

        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)

        category = category or random.choice(list(category_wise_examples.keys()))
        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )
        subj = (
            random.choice(list(category_wise_examples[category].values))
            if subj is None
            else subj
        )

        options = random.sample(
            list(category_wise_examples[category].values), n_options
        )
        # print(f"{category=}")
        # print(f"{subj=}")
        # print(f"{options=}")

        metadata = {}

        sample = SelectAllSample(
            subj=subj,
            options=options,
            category=category,
            metadata=metadata,
            prompt_template=self.prompt_templates[prompt_template_idx],
        )

        if filter_by_lm_prediction:
            if retry_count > 10:
                return
            prompt = sample.prompt()
            inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = inputs.data

            predictions = predict_next_token(
                mt=mt,
                inputs=inputs,
            )[0]
            all_token_id = mt.tokenizer.encode(" All", add_special_tokens=False)[0]
            if predictions[0].token_id != all_token_id:
                logger.error(
                    f"""Sample = {sample}
    Top prediction {predictions[0]} does not match the expected answer " All".
    Retry count: {retry_count + 1}. Retrying ...
    """
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    category=category,
                    subj=subj,
                    n_options=n_options,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    retry_count=retry_count + 1,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample


@dataclass
class CountingTask(DataClassJsonMixin):
    category_type: str
    prompt_templates: list[str]
    category_wise_examples: dict[str, list] = field(default_factory=dict)
    task_name: str = "counting"
    exclude_categories: dict[str, list] = field(default_factory=dict)

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            counting_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(counting_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return CountingTask(
                task_name="counting",
                category_type=data.get("name", category_type),
                prompt_templates=data["count_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
                exclude_categories=data.get("exclude_categories", {}),
            )

    @property
    def categories(self):
        """
        Returns the list of categories in the task.
        """
        return list(self.category_wise_examples.keys())

    def exclude_for_category(self, category):
        if category in self.exclude_categories:
            return self.exclude_categories[category]
        else:
            return []

    def __str__(self):
        return f"""CountingTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}"""

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        n_options: int = 2,
        n_distractors: int = 3,
        filter_by_lm_prediction: bool = False,
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
    ) -> CountingSample:
        """
        Get a random sample with the specified attribute.
        """

        kwargs = {  # noqa
            "prompt_template_idx": prompt_template_idx,
            "option_style": option_style,
        }
        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)

        category = category or random.choice(list(category_wise_examples.keys()))
        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )

        counting_items = random.sample(
            category_wise_examples[category].values, k=n_options
        )

        distractors = []
        while len(distractors) < n_distractors:
            other_category = random.choice(
                list(
                    set(category_wise_examples.keys())
                    - set([category] + exclude_distractor_categories)
                )
            )
            distractors.append(
                random.choice(
                    (
                        category_wise_examples[other_category]
                        - KeyedSet(
                            counting_items + exclude_objs + distractors,
                            tokenizer=tokenizer,
                        )
                    ).values
                )
            )

        options = counting_items + distractors[:n_distractors]
        random.shuffle(options)

        sample = CountingSample(
            prompt_template=self.prompt_templates[prompt_template_idx],
            options=options,
            count=n_options,
            category=category,
            default_option_style=option_style,
            ans_token_id=get_first_token_id(
                name=COUNT_STR_MAP[n_options], tokenizer=tokenizer, prefix=" "
            ),
        )

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                    for opt in get_options_for_answer(sample)
                ],
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {sample.ans_token_id}, "{mt.tokenizer.decode(sample.ans_token_id)}".
Retry count: {retry_count + 1}. Retrying ..."""
                )
                return self.get_random_sample(
                    mt=mt,
                    n_options=n_options,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    category=category,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    **kwargs,
                )

            sample.prediction = [pred for token_id, (rank, pred) in track_objs.items()]

        sample.metadata["retry_count"] = retry_count
        return sample


@dataclass
class DeductionTask(DataClassJsonMixin):
    topics: dict[str, dict]
    logic_templates: dict[str, list]
    task_name: str = "deduction"

    @staticmethod
    def load(dir_path: Path | None = os.path.join(DEFAULT_DATA_DIR, "deduction")):
        for file_path in Path(dir_path).iterdir():
            if file_path.is_file() and file_path.name == "topics.json":
                print(file_path)
                with file_path.open("r") as f:
                    topics = json.load(f)
            elif file_path.is_file() and file_path.name == "logic_templates.json":
                with file_path.open("r") as f:
                    logic_templates = json.load(f)

        return DeductionTask(
            task_name="deduction",
            topics=topics,
            logic_templates=logic_templates,
        )

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        topic_name: str,
        depth: int,
        names: list[str] = ["Alice", "Bob", "Cam", "Dave", "Eli"],
        filter_by_lm_prediction: bool = True,
        retry_count: int = 0,
    ) -> DeductionSample:
        """
        Get a random sample from the logical deduction problems.
        """

        assert topic_name in list(
            self.topics.keys()
        ), f"Topic name '{topic_name}' not a valid topic: {list(self.topics.keys())}."
        assert str(depth) in list(
            self.logic_templates.keys()
        ), f"Depth '{depth}' not a valid depth: {list(self.logic_templates.keys())}."
        assert (
            len(names) >= depth
        ), f"Length of names list must be greater than depth '{depth}'."

        tokenizer = unwrap_tokenizer(mt)

        # Select components
        topic_vocab = self.topics[topic_name]
        template = random.choice(self.logic_templates[str(depth)])

        # Build the premise sentences
        premise_sentences = []
        for p in literal_eval(template["premises"]):
            idx1, comparator, idx2 = p
            name1 = names[idx1]
            name2 = names[idx2]

            if comparator == ">":
                comparator_text = topic_vocab["positive_comparator"]
            else:
                comparator_text = topic_vocab["negative_comparator"]

            premise_sentences.append(f"{name1} is {comparator_text} {name2}.")

        # Build final question
        if template["question_type"] == "max":
            question_word = topic_vocab["positive_extreme"]
        else:
            question_word = topic_vocab["negative_extreme"]

        question_text = f"Who is the {question_word}?"

        # Combine and determine answer
        full_question = " ".join(premise_sentences) + f" {question_text}"
        answer = names[template["answer_idx"]]
        final_prompt = "Answer with the name only. " + full_question

        metadata = {}

        sample = DeductionSample(
            prompt=final_prompt,
            answer=answer,
            depth=depth,
            topic=topic_name,
            prediction=None,
            metadata=metadata,
        )

        if filter_by_lm_prediction:
            prompt = sample.prompt
            inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = inputs.data

            predictions = predict_next_token(
                mt=mt,
                inputs=inputs,
            )[0]
            answer_token_id = get_first_token_id(answer, tokenizer, prefix=" ")
            if predictions[0].token_id != answer_token_id:
                logger.error(
                    f"""Sample = {sample}
                    Top prediction {predictions[0]} does not match the answer '{answer}'.
                    Retry count: {retry_count + 1}. Retrying ...
                    """
                )
                return self.get_random_sample(
                    mt=mt,
                    topic_name=topic_name,
                    depth=depth,
                    names=names,
                    filter_by_lm_prediction=True,
                    retry_count=retry_count + 1,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample

    def __str__(self):
        return f"""DeductionTask
Depths: {list(self.logic_templates.keys())}
Topics: {list(self.topics.keys())}
        """


@dataclass
class YesNoTask(DataClassJsonMixin):
    category_type: str
    prompt_templates: list[str]
    category_wise_examples: dict[str, list] = field(default_factory=dict)
    task_name: str = "yes_no"
    exclude_categories: dict[str, list] = field(default_factory=dict)

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            counting_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(counting_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return YesNoTask(
                task_name="yes_no",
                category_type=data.get("name", category_type),
                prompt_templates=data["yes_no_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
                exclude_categories=data.get("exclude_categories", {}),
            )

    def exclude_for_category(self, category):
        if category in self.exclude_categories:
            return self.exclude_categories[category]
        else:
            return []

    @property
    def categories(self):
        """
        Returns the list of categories in the task.
        """
        return list(self.category_wise_examples.keys())

    def __str__(self):
        return f"""YesNoTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}"""

    def get_random_sample(
        self,
        mt: ModelandTokenizer = None,
        prompt_template_idx: int = 0,
        yes_mode: bool | None = None,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        n_options: int = 5,
        filter_by_lm_prediction: bool = False,
        retry_count: int = 0,
    ) -> YesNoSample:
        """
        Get a random sample with a specified attribute.
        """

        tokenizer = unwrap_tokenizer(mt)

        category_wise_examples = {}
        for cat in self.category_wise_examples:
            examples = copy.deepcopy(self.category_wise_examples[cat])
            random.shuffle(examples)
            category_wise_examples[cat] = KeyedSet(examples, tokenizer=tokenizer)

        category = category or random.choice(self.categories)
        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )

        yes_mode = random.choice([True, False]) if yes_mode is None else yes_mode

        options = []
        if yes_mode:
            valid_options = random.sample(
                category_wise_examples[category].values,
                k=random.choice(range(1, n_options - 1)),
            )
            options.extend(valid_options)

        n_distractors = n_options - len(options)
        for _ in range(n_distractors):
            other_category = random.choice(
                list(
                    set(category_wise_examples.keys())
                    - set([category] + self.exclude_for_category(category))
                )
            )
            options.append(random.choice(category_wise_examples[other_category].values))

        random.shuffle(options)

        sample = YesNoSample(
            prompt_template=self.prompt_templates[prompt_template_idx],
            options=options,
            yes=yes_mode,
            category=category,
            prediction=None,
            default_option_style=option_style,
            ans_token_id=get_first_token_id(
                name="Yes" if yes_mode else "No", tokenizer=tokenizer, prefix=" "
            ),
        )

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                    for opt in get_options_for_answer(sample)
                ],
                prefix=" ",
            )

            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {sample.ans_token_id}, "{mt.tokenizer.decode(sample.ans_token_id)}".
Retry count: {retry_count + 1}. Retrying ..."""
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    yes_mode=yes_mode,
                    option_style=option_style,
                    category=category,
                    n_options=n_options,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    retry_count=retry_count + 1,
                )

            sample.prediction = [pred for token_id, (rank, pred) in track_objs.items()]

        sample.metadata["retry_count"] = retry_count
        return sample


@dataclass
class SelectFirstTask(SelectOneTask):
    task_name: str = "select_first"

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return SelectFirstTask(
                task_name="select_first",
                category_type=data.get("name", category_type),
                prompt_templates=data["first_item_in_cat_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
                exclude_categories=data.get("exclude_categories", {}),
            )

    def __str__(self):
        return f"""SelectFirstTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        subj: str | None = None,
        n_distractors: int = 5,
        filter_by_lm_prediction: bool = False,
        obj_idx: int | None = None,
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
        output_formatting: Literal["zero_shot", "object", "lettered"] = "zero_shot",
    ) -> SelectionSample:

        sample = super().get_random_sample(
            mt=mt,
            prompt_template_idx=0,
            option_style=option_style,
            category=category,
            subj=subj,
            n_distractors=n_distractors,
            filter_by_lm_prediction=False,  # disable filtering in the parent call
            obj_idx=obj_idx,
            exclude_objs=exclude_objs,
            exclude_distractor_categories=exclude_distractor_categories,
            insert_distractor=insert_distractor,
            retry_count=retry_count,
            output_formatting=output_formatting,
        )

        # add other items of the same category
        category_items = copy.deepcopy(self.category_wise_examples[sample.category])
        random.shuffle(category_items)
        category_items = KeyedSet(category_items, mt.tokenizer)
        num_additional_items = random.randint(1, 3)
        additional_items = random.sample(
            (
                category_items - KeyedSet(sample.options + exclude_objs, mt.tokenizer)
            ).values,
            k=num_additional_items,
        )
        sample.options = sample.options + additional_items
        random.shuffle(sample.options)

        category_items = [sample.obj] + additional_items
        obj, obj_idx = None, None
        for idx, option in enumerate(sample.options):
            if option in category_items:
                if obj is None:
                    obj = option
                    obj_idx = idx
                break

        sample.obj = obj
        sample.obj_idx = obj_idx
        sample.ans_token_id = get_first_token_id(obj, mt.tokenizer, prefix=" ")
        sample.prompt_template = self.prompt_templates[prompt_template_idx]
        sample.default_option_style = option_style

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                    for opt in get_options_for_answer(sample)
                ],
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {sample.ans_token_id}, "{mt.tokenizer.decode(sample.ans_token_id)}".
Retry count: {retry_count + 1}. Retrying ..."""
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    category=category,
                    subj=subj,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    obj_idx=obj_idx,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    output_formatting=output_formatting,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample


@dataclass
class SelectLastTask(SelectOneTask):
    task_name: str = "select_last"

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "selection/objects.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            selection_root = os.path.join(DEFAULT_DATA_DIR, "selection")
            path = os.path.join(selection_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return SelectLastTask(
                task_name="select_last",
                category_type=data.get("name", category_type),
                prompt_templates=data["last_item_in_cat_prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
                exclude_categories=data.get("exclude_categories", {}),
            )

    def __str__(self):
        return f"""SelectLastTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        subj: str | None = None,
        n_distractors: int = 5,
        filter_by_lm_prediction: bool = False,
        obj_idx: int | None = None,
        exclude_objs: Sequence[str] = [],
        exclude_distractor_categories: Sequence[str] = [],
        insert_distractor: Sequence[tuple[str, int]] = [],
        retry_count: int = 0,
        output_formatting: Literal["zero_shot", "object", "lettered"] = "zero_shot",
    ) -> SelectionSample:

        sample = super().get_random_sample(
            mt=mt,
            prompt_template_idx=0,
            option_style=option_style,
            category=category,
            subj=subj,
            n_distractors=n_distractors,
            filter_by_lm_prediction=False,  # disable filtering in the parent call
            obj_idx=obj_idx,
            exclude_objs=exclude_objs,
            exclude_distractor_categories=exclude_distractor_categories,
            insert_distractor=insert_distractor,
            retry_count=retry_count,
            output_formatting=output_formatting,
        )

        # add other items of the same category
        category_items = copy.deepcopy(self.category_wise_examples[sample.category])
        random.shuffle(category_items)
        category_items = KeyedSet(category_items, mt.tokenizer)
        num_additional_items = random.randint(1, 3)
        additional_items = random.sample(
            (
                category_items - KeyedSet(sample.options + exclude_objs, mt.tokenizer)
            ).values,
            k=num_additional_items,
        )
        sample.options = sample.options + additional_items
        random.shuffle(sample.options)

        category_items = [sample.obj] + additional_items
        obj, obj_idx = None, None
        for idx, option in enumerate(sample.options):
            if option in category_items:
                obj = option
                obj_idx = idx
                # break # just removing break to get the last item

        assert (
            obj is not None and obj_idx is not None
        ), "Failed to find the last item in the options."

        sample.obj = obj
        sample.obj_idx = obj_idx
        sample.ans_token_id = get_first_token_id(obj, mt.tokenizer, prefix=" ")
        sample.prompt_template = self.prompt_templates[prompt_template_idx]
        sample.default_option_style = option_style

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            # logger.info(f"\nPrompt: {prompt}")
            tokenized_inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = tokenized_inputs.data

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=tokenized_inputs,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                    for opt in get_options_for_answer(sample)
                ],
                prefix=" ",
            )

            # if predictions[0].token_id != obj_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
Top prediction {track_objs[list(track_objs.keys())[0]]} does not match the object {sample.ans_token_id}, "{mt.tokenizer.decode(sample.ans_token_id)}".
Retry count: {retry_count + 1}. Retrying ..."""
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    category=category,
                    subj=subj,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    obj_idx=obj_idx,
                    exclude_objs=exclude_objs,
                    exclude_distractor_categories=exclude_distractor_categories,
                    insert_distractor=insert_distractor,
                    retry_count=retry_count + 1,
                    output_formatting=output_formatting,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample


########################################### </Task Data Classes> ##########################################


#################################################################################################
# Counterfactual Sample Generation for patching experiments
#################################################################################################
def get_options_for_answer(sample: SelectionSample | CountingSample | YesNoSample):
    if isinstance(sample, SelectionSample):
        if sample.metadata.get("question_type", "obj") == "MCQ":
            return [
                chr(ord(sample.option_label_start_from) + i)
                for i in range(len(sample.options))
            ]
        return sample.options
    elif isinstance(sample, CountingSample):
        return list(COUNT_STR_MAP.values())
    elif isinstance(sample, YesNoSample):
        return ["Yes", "No"]
    else:
        raise ValueError(f"Unsupported sample type: {type(sample)}")


@torch.inference_mode()
def get_counterfactual_samples_within_task(
    task: SelectOneTask | SelectOrderTask,
    mt: ModelandTokenizer,
    patch_category: str | None = None,
    clean_category: str | None = None,
    shuffle_clean_options: bool = False,
    filter_by_lm_prediction: bool = True,
    distinct_options: bool = True,
    n_distractors: int = 5,
    patch_n_distractors: int | None = None,
    clean_n_distractors: int | None = None,
    prompt_template_idx=3,
    patch_prompt_template_idx: int | None = None,
    clean_prompt_template_idx: int | None = None,
    option_style="single_line",
    patch_option_style: str | None = None,
    clean_option_style: str | None = None,
    mcqify: bool = False,
):
    categories = list(task.category_wise_examples.keys())
    if patch_category is None:
        patch_category = random.choice(categories)
    if patch_n_distractors is None:
        patch_n_distractors = n_distractors
    if patch_prompt_template_idx is None:
        patch_prompt_template_idx = prompt_template_idx
    if clean_prompt_template_idx is None:
        clean_prompt_template_idx = prompt_template_idx

    patch_subj, patch_obj = random.sample(
        task.category_wise_examples[patch_category], 2
    )
    # logger.info(
    #     f"Patch category: {patch_category}, subject: {patch_subj}, object: {patch_obj}"
    # )

    if clean_category is None:
        clean_category = random.choice(
            list(
                set(categories)
                - {patch_category}
                - set(task.exclude_for_category(patch_category))
            )
        )

    clean_options = task.category_wise_examples[clean_category]
    random.shuffle(clean_options)

    clean_subj, clean_obj = random.sample(
        (
            KeyedSet(clean_options, mt.tokenizer) - KeyedSet([patch_obj], mt.tokenizer)
        ).values,
        2,
    )
    # logger.info(
    #     f"Clean category: {clean_category}, subject: {clean_subj}, object: {clean_obj}"
    # )

    if distinct_options is False:
        patch_type_obj = patch_obj
        clean_type_obj = clean_obj
    else:
        patch_type_obj = random.choice(
            (
                KeyedSet(task.category_wise_examples[patch_category], mt.tokenizer)
                - KeyedSet([patch_obj], mt.tokenizer)
            ).values
        )
        clean_type_obj = random.choice(
            (
                KeyedSet(task.category_wise_examples[clean_category], mt.tokenizer)
                - KeyedSet([clean_obj], mt.tokenizer)
            ).values
        )

    patch_must_have_options = [patch_obj, clean_type_obj]
    clean_must_have_options = [clean_obj, patch_type_obj]

    # logger.info(f"{patch_must_have_options=}")
    # logger.info(f"{clean_must_have_options=}")
    # logger.info(f"{clean_type_obj=}")
    # logger.info(f"{patch_type_obj=}")

    patch_distractors = []
    other_categories = random.sample(
        list(
            set(categories)
            - (
                {patch_category, clean_category}
                | set(task.exclude_for_category(clean_category))
                | set(
                    task.exclude_for_category(patch_category)
                )  # TODO (arnab): actually do this for all sampling
            )
        ),
        k=patch_n_distractors - (len(patch_must_have_options)) + 1,
    )

    for other_category in other_categories:
        other_examples = task.category_wise_examples[other_category]
        random.shuffle(other_examples)
        other_examples = KeyedSet(other_examples, mt.tokenizer)
        patch_distractors.append(
            random.choice(
                (
                    other_examples
                    - KeyedSet(
                        patch_must_have_options + patch_distractors,
                        tokenizer=mt.tokenizer,
                    )
                ).values
            )
        )

    patch_options = patch_must_have_options + patch_distractors
    random.shuffle(patch_options)
    patch_obj_idx = patch_options.index(patch_obj)
    # logger.info(f"{patch_obj_idx=} | {patch_options}")

    if distinct_options is not True:
        if clean_n_distractors is not None:
            logger.warning(
                f"Passed {clean_n_distractors=}. But distinct_options is False, so clean options will be same as patch options."
            )
        clean_options = copy.deepcopy(patch_options)
        if shuffle_clean_options:
            # Useful for the pointer experiments
            while (
                clean_options.index(clean_obj) == patch_obj_idx
                or clean_options.index(patch_type_obj) == patch_obj_idx
            ):
                random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    else:
        if clean_n_distractors is None:
            clean_n_distractors = n_distractors
        other_categories = random.sample(
            list(
                set(categories)
                - (
                    {patch_category, clean_category}
                    | set(task.exclude_for_category(clean_category))
                    | set(task.exclude_for_category(patch_category))
                )
            ),
            k=clean_n_distractors - (len(clean_must_have_options)) + 1,
        )
        clean_distractors = []
        for other_category in other_categories:
            other_examples = task.category_wise_examples[other_category]
            random.shuffle(other_examples)
            other_examples = KeyedSet(other_examples, mt.tokenizer)
            clean_distractors.append(
                random.choice(
                    (
                        other_examples
                        - KeyedSet(
                            clean_must_have_options + clean_distractors,
                            tokenizer=mt.tokenizer,
                        )
                    ).values
                )
            )
        clean_options = clean_must_have_options + clean_distractors
        random.shuffle(clean_options)
        while (
            clean_options.index(clean_obj) == patch_obj_idx
            or clean_options.index(patch_type_obj) == patch_obj_idx
        ):
            random.shuffle(clean_options)
        clean_obj_idx = clean_options.index(clean_obj)

    logger.info(f"{clean_obj_idx=} | {clean_options}")

    print(f"{type(task)=}")
    if isinstance(task, SelectOrderTask):
        patch_metadata = {
            "track_type_obj_idx": clean_obj_idx,
            "track_type_obj": patch_options[clean_obj_idx],
            "track_type_obj_token_id": get_first_token_id(
                patch_options[clean_obj_idx], mt.tokenizer, prefix=" "
            ),
        }
        clean_metadata = {
            "track_type_obj_idx": patch_obj_idx,
            "track_type_obj": clean_options[patch_obj_idx],
            "track_type_obj_token_id": get_first_token_id(
                clean_options[patch_obj_idx], mt.tokenizer, prefix=" "
            ),
        }
    elif isinstance(task, SelectOneTask):
        patch_metadata = {
            "track_category": clean_category,
            "track_type_obj": clean_type_obj,
            "track_type_obj_idx": patch_options.index(clean_type_obj),
            "track_type_obj_token_id": get_first_token_id(
                clean_type_obj, mt.tokenizer, prefix=" "
            ),
        }
        clean_metadata = {
            "track_category": patch_category,
            "track_type_obj": patch_type_obj,
            "track_type_obj_idx": clean_options.index(patch_type_obj),
            "track_type_obj_token_id": get_first_token_id(
                patch_type_obj, mt.tokenizer, prefix=" "
            ),
        }
    else:
        raise NotImplementedError(f"Unsupported task type: {type(task)}")

    patch_sample = SelectionSample(
        subj=patch_subj,
        obj=patch_obj,
        answer=patch_obj,
        obj_idx=patch_obj_idx,
        ans_token_id=get_first_token_id(patch_obj, mt.tokenizer, prefix=" "),
        options=patch_options,
        category=patch_category,
        metadata=patch_metadata,
        prompt_template=task.prompt_templates[patch_prompt_template_idx],
        default_option_style=patch_option_style or option_style,
    )
    clean_sample = SelectionSample(
        subj=clean_subj,
        obj=clean_obj,
        answer=clean_obj,
        obj_idx=clean_obj_idx,
        ans_token_id=get_first_token_id(clean_obj, mt.tokenizer, prefix=" "),
        options=clean_options,
        category=clean_category,
        metadata=clean_metadata,
        prompt_template=task.prompt_templates[clean_prompt_template_idx],
        default_option_style=clean_option_style or option_style,
    )

    if mcqify:
        start_options = ["a", "p"]
        patch_start = random.choice(start_options)
        clean_start = random.choice(list(set(start_options) - {patch_start}))
        patch_sample = MCQify_sample(
            tokenizer=mt.tokenizer, sample=patch_sample, start_from=patch_start
        )
        clean_sample = MCQify_sample(
            tokenizer=mt.tokenizer, sample=clean_sample, start_from=clean_start
        )

        target_obj_idx = clean_sample.metadata["track_type_obj_idx"]
        clean_sample.metadata["track_type_obj_token_id"] = get_first_token_id(
            name=chr(ord(clean_start) + target_obj_idx),
            tokenizer=mt.tokenizer,
            prefix=" ",
        )

    if "qwen" in mt.name.lower():
        #! for attention sink in qwen models
        patch_sample.prompt_template = "# " + patch_sample.prompt_template
        clean_sample.prompt_template = "# " + clean_sample.prompt_template

    if filter_by_lm_prediction:
        test_samples = [patch_sample, clean_sample]
        if distinct_options is True:
            clean_sample_2 = copy.deepcopy(patch_sample)
            clean_sample_2.options = clean_options
            clean_sample_2.obj = clean_sample.metadata["track_type_obj"]
            clean_sample_2.obj_idx = clean_sample.metadata["track_type_obj_idx"]
            if not mcqify:
                clean_sample_2.ans_token_id = clean_sample.metadata[
                    "track_type_obj_token_id"
                ]
            else:
                clean_sample_2.ans_token_id = get_first_token_id(
                    name=chr(
                        ord(clean_sample_2.option_label_start_from)
                        + clean_sample_2.obj_idx
                    ),
                    tokenizer=mt.tokenizer,
                    prefix=" ",
                )
            test_samples.append(clean_sample_2)

        for sample in test_samples:
            tokenized = prepare_input(tokenizer=mt, prompts=sample.prompt())
            is_correct, predictions, track_options = verify_correct_option(
                mt=mt,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(opt, mt.tokenizer, prefix=" ")
                    for opt in get_options_for_answer(sample)
                ],
                input=tokenized,
            )
            # sample.metadata["tokenized"] = tokenized.data
            logger.info(sample.prompt())
            logger.info(
                f"{sample.subj} | {sample.category} -> {sample.obj} | pred={[str(p) for p in predictions]}"
            )
            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                # return ValueError("Testing")
                return get_counterfactual_samples_within_task(
                    task=task,
                    mt=mt,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    shuffle_clean_options=shuffle_clean_options,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    distinct_options=distinct_options,
                    n_distractors=n_distractors,
                    patch_n_distractors=patch_n_distractors,
                    clean_n_distractors=clean_n_distractors,
                    prompt_template_idx=prompt_template_idx,
                    patch_prompt_template_idx=patch_prompt_template_idx,
                    clean_prompt_template_idx=clean_prompt_template_idx,
                    option_style=option_style,
                    patch_option_style=patch_option_style,
                    clean_option_style=clean_option_style,
                    mcqify=mcqify,
                )
            sample.prediction = predictions

    # find the "?" token position in the samples
    for sample in [patch_sample, clean_sample]:
        tokenized = prepare_input(
            tokenizer=mt, prompts=sample.prompt(), return_offsets_mapping=True
        )
        offsets = tokenized.pop("offset_mapping")[0]
        ques_range = find_token_range(
            string=sample.prompt(),
            substring="?",
            tokenizer=mt.tokenizer,
            offset_mapping=offsets,
        )
        sample.metadata["ques_pos"] = ques_range[1] - 1
        sample.metadata["tokenized"] = tokenized.data

    return patch_sample, clean_sample


@torch.inference_mode()
def get_counterfactual_samples_within_counting_task(
    task: CountingTask,
    mt: ModelandTokenizer,
    patch_category=None,
    clean_category=None,
    filter_by_lm_prediction=True,
    distinct_options: bool = True,
    n_options: int = 5,
    clean_n_options: int | None = None,
    patch_n_options: int | None = None,
    prompt_template_idx=1,
    patch_prompt_template_idx: int | None = None,
    clean_prompt_template_idx: int | None = None,
    option_style="single_line",
    patch_option_style: str | None = None,
    clean_option_style: str | None = None,
    verbose=False,
    retry_count=0,
):
    # Set parameter defaults
    if patch_prompt_template_idx is None:
        patch_prompt_template_idx = prompt_template_idx
    if clean_prompt_template_idx is None:
        clean_prompt_template_idx = prompt_template_idx

    categories = list(task.category_wise_examples.keys())
    patch_category = patch_category or random.choice(categories)
    clean_category = clean_category or random.choice(
        list(
            set(categories)
            - set([patch_category] + task.exclude_for_category(patch_category))
        )
    )

    clean_n_options = clean_n_options or n_options
    patch_n_options = patch_n_options or n_options

    def divide_cat_vs_distractors(total: int) -> tuple[int, int]:
        assert total > 2, "total must be greater than 2"
        while True:
            n_cat = random.randint(1, total - 1)
            n_distractors = total - n_cat
            if n_cat != n_distractors:
                break
        return n_cat, n_distractors

    def get_counting_sample(
        n_cat_clean: int,
        n_cat_patch: int,
        clean_cat: str = clean_category,
        patch_cat: str = patch_category,
        exclude_objs: list[str] = [],
    ) -> CountingSample:
        clean_objs = random.sample(
            (
                KeyedSet(task.category_wise_examples[clean_cat], mt.tokenizer)
                - KeyedSet(exclude_objs, mt.tokenizer)
            ).values,
            n_cat_clean,
        )
        patch_objs = random.sample(
            (
                KeyedSet(task.category_wise_examples[patch_cat], mt.tokenizer)
                - KeyedSet(exclude_objs + clean_objs, mt.tokenizer)
            ).values,
            n_cat_patch,
        )
        options = clean_objs + patch_objs
        random.shuffle(options)
        return CountingSample(
            options=options,
            count=len(clean_objs),
            category=clean_cat,
            prompt_template=task.prompt_templates[clean_prompt_template_idx],
            default_option_style=clean_option_style or option_style,
            ans_token_id=get_first_token_id(
                name=COUNT_STR_MAP[n_cat_clean], tokenizer=mt.tokenizer, prefix=" "
            ),
        )

    if distinct_options is False:
        assert clean_n_options == patch_n_options
        n_cat_clean, n_cat_patch = divide_cat_vs_distractors(clean_n_options)
        clean_sample = get_counting_sample(n_cat_clean, n_cat_patch)

        patch_sample = copy.deepcopy(clean_sample)
        patch_sample.count = n_cat_patch
        patch_sample.category = patch_category
        patch_sample.prompt_template = task.prompt_templates[patch_prompt_template_idx]
        patch_sample.ans_token_id = get_first_token_id(
            name=COUNT_STR_MAP[n_cat_patch], tokenizer=mt.tokenizer, prefix=" "
        )

        clean_sample.metadata = {
            "track_category": patch_category,
            "track_count": n_cat_patch,
            "track_type_obj_token_id": get_first_token_id(
                name=COUNT_STR_MAP[n_cat_patch], tokenizer=mt.tokenizer, prefix=" "
            ),
        }
    else:
        while True:
            n_clean_cat_clean, n_clean_cat_patch = divide_cat_vs_distractors(
                clean_n_options
            )
            n_patch_cat_patch, n_patch_cat_clean = divide_cat_vs_distractors(
                patch_n_options
            )
            if (
                n_clean_cat_patch != n_patch_cat_patch
            ):  #! ensures different counts of patch category
                break

        clean_sample = get_counting_sample(
            n_cat_clean=n_clean_cat_clean,
            n_cat_patch=n_clean_cat_patch,
            clean_cat=clean_category,
            patch_cat=patch_category,
        )
        patch_sample = get_counting_sample(
            n_cat_clean=n_patch_cat_patch,
            n_cat_patch=n_patch_cat_clean,
            clean_cat=patch_category,
            patch_cat=clean_category,
            exclude_objs=clean_sample.options,
        )

        clean_sample.metadata = {
            "track_category": patch_category,
            "track_count": n_clean_cat_patch,
            "track_type_obj_token_id": get_first_token_id(
                name=COUNT_STR_MAP[n_clean_cat_patch],
                tokenizer=mt.tokenizer,
                prefix=" ",
            ),
        }

    logger.debug(f"{clean_category=} | {clean_sample.options=}")
    logger.debug(f"{patch_category=} | {patch_sample.options=}")

    if filter_by_lm_prediction:
        test_samples = [patch_sample, clean_sample]
        if distinct_options is True:
            gold_sample = copy.deepcopy(patch_sample)
            gold_sample.options = clean_sample.options
            gold_sample.ans_token_id = clean_sample.metadata["track_type_obj_token_id"]
            gold_sample.count = clean_sample.metadata["track_count"]
            test_samples.append(gold_sample)

        for sample in test_samples:
            logger.info(
                f"{sample.prompt()} >> {mt.tokenizer.decode(sample.ans_token_id)}"
            )
            if retry_count >= 10:
                break
            tokenized = prepare_input(tokenizer=mt, prompts=sample.prompt())
            target_token_id = get_first_token_id(
                name=COUNT_STR_MAP[sample.count], tokenizer=mt.tokenizer, prefix=" "
            )
            option_token_ids = [
                get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                for opt in get_options_for_answer(sample)
            ]
            print(f"{target_token_id=} | {mt.tokenizer.decode(target_token_id)=}")
            is_correct, predictions, track_options = verify_correct_option(
                mt=mt,
                target=target_token_id,
                options=option_token_ids,
                input=tokenized,
            )
            sample.metadata["tokenized"] = tokenized.data

            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                return get_counterfactual_samples_within_counting_task(
                    task=task,
                    mt=mt,
                    n_options=n_options,
                    clean_n_options=clean_n_options,
                    patch_n_options=patch_n_options,
                    prompt_template_idx=prompt_template_idx,
                    patch_prompt_template_idx=patch_prompt_template_idx,
                    clean_prompt_template_idx=clean_prompt_template_idx,
                    option_style=option_style,
                    patch_option_style=patch_option_style,
                    clean_option_style=clean_option_style,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    verbose=verbose,
                    retry_count=retry_count + 1,
                    distinct_options=distinct_options,
                )
            sample.prediction = [
                pred for token_id, (rank, pred) in track_options.items()
            ]

    # find the "?" token position in the samples
    for sample in [patch_sample, clean_sample]:
        tokenized = prepare_input(
            tokenizer=mt, prompts=sample.prompt(), return_offsets_mapping=True
        )
        offsets = tokenized.pop("offset_mapping")[0]
        ques_range = find_token_range(
            string=sample.prompt(),
            substring="?",
            tokenizer=mt.tokenizer,
            offset_mapping=offsets,
        )
        sample.metadata["ques_pos"] = ques_range[1] - 1
        sample.metadata["tokenized"] = tokenized.data

    return patch_sample, clean_sample


@torch.inference_mode()
def get_counterfactual_samples_within_yes_no_task(
    task: YesNoTask,
    mt: ModelandTokenizer,
    patch_category=None,
    clean_category=None,
    filter_by_lm_prediction=True,
    n_options: int = 5,
    clean_n_options: int | None = None,
    patch_n_options: int | None = None,
    prompt_template_idx=1,
    patch_prompt_template_idx: int | None = None,
    clean_prompt_template_idx: int | None = None,
    option_style="single_line",
    patch_option_style: str | None = None,
    clean_option_style: str | None = None,
    verbose=False,
    patch_yes_mode: bool | None = None,
    retry_count=0,
):
    # Set parameter defaults
    patch_prompt_template_idx = patch_prompt_template_idx or prompt_template_idx
    clean_prompt_template_idx = clean_prompt_template_idx or prompt_template_idx

    patch_option_style = patch_option_style or option_style
    clean_option_style = clean_option_style or option_style

    categories = list(task.category_wise_examples.keys())
    patch_category = patch_category or random.choice(categories)
    clean_category = clean_category or random.choice(
        list(
            set(categories)
            - set([patch_category] + task.exclude_for_category(patch_category))
        )
    )

    clean_n_options = clean_n_options or n_options
    patch_n_options = patch_n_options or n_options

    patch_yes_mode = (
        patch_yes_mode if patch_yes_mode is not None else random.choice([True, False])
    )

    def get_yes_no_sample(
        category: str,
        n_options: int,
        yes_mode: bool,
        exclude_objs: list[str] = [],
        exclude_distractor_categories: list[str] = [],
        must_have_distractor_category: str | None = None,
    ):
        options = []
        if yes_mode:
            valid_options = random.sample(
                (
                    KeyedSet(task.category_wise_examples[category], mt.tokenizer)
                    - KeyedSet(exclude_objs, mt.tokenizer)
                ).values,
                k=random.choice(range(1, n_options - 1)),
            )
            options.extend(valid_options)

        n_distractors = n_options - len(options)
        if must_have_distractor_category is not None:
            assert n_distractors >= 1
            other_category = must_have_distractor_category
            options.append(
                random.choice(
                    (
                        KeyedSet(
                            task.category_wise_examples[other_category], mt.tokenizer
                        )
                        - KeyedSet(exclude_objs + options, mt.tokenizer)
                    ).values
                )
            )
            n_distractors -= 1
        other_categories = list(
            set(task.category_wise_examples.keys())
            - {category}
            - set(exclude_distractor_categories)
        )
        for _ in range(n_distractors):
            other_category = random.choice(other_categories)
            options.append(
                random.choice(
                    (
                        KeyedSet(
                            task.category_wise_examples[other_category], mt.tokenizer
                        )
                        - KeyedSet(exclude_objs + options, mt.tokenizer)
                    ).values
                )
            )

        random.shuffle(options)

        return YesNoSample(
            prompt_template=task.prompt_templates[prompt_template_idx],
            options=options,
            yes=yes_mode,
            category=category,
            prediction=None,
            default_option_style=option_style,
            ans_token_id=get_first_token_id(
                name="Yes" if yes_mode else "No", tokenizer=mt.tokenizer, prefix=" "
            ),
        )

    patch_sample = get_yes_no_sample(
        category=patch_category,
        n_options=patch_n_options,
        yes_mode=patch_yes_mode,
        exclude_objs=[],
        exclude_distractor_categories=[],
    )

    # to include or not to include a patch category item
    patch_cat_kwargs = {
        "must_have_distractor_category": None,
        "exclude_distractor_categories": [],
    }
    if not patch_yes_mode:
        # patch ans is No, so intervention answer should be yes
        patch_cat_kwargs["must_have_distractor_category"] = patch_category
    else:
        # patch ans is Yes, so intervention answer should be no
        patch_cat_kwargs["exclude_distractor_categories"] = [patch_category]
    clean_sample = get_yes_no_sample(
        category=clean_category,
        n_options=clean_n_options,
        yes_mode=patch_yes_mode,
        exclude_objs=patch_sample.options,
        **patch_cat_kwargs,
    )

    clean_sample.metadata = {
        "track_category": patch_category,
        "track_yes_mode": not patch_yes_mode,  # the answer should change
        "track_type_obj_token_id": get_first_token_id(
            name="Yes" if not patch_yes_mode else "No",
            tokenizer=mt.tokenizer,
            prefix=" ",
        ),
    }

    if filter_by_lm_prediction:
        test_samples = [patch_sample, clean_sample]
        gold_sample = copy.deepcopy(patch_sample)
        gold_sample.options = clean_sample.options
        gold_sample.yes = clean_sample.metadata["track_yes_mode"]
        gold_sample.ans_token_id = clean_sample.metadata["track_type_obj_token_id"]
        test_samples.append(gold_sample)

        for sample in test_samples:
            logger.info(
                f"{sample.prompt()} >> {mt.tokenizer.decode(sample.ans_token_id)}"
            )
            if retry_count >= 10:
                break
            tokenized = prepare_input(tokenizer=mt, prompts=sample.prompt())
            is_correct, predictions, track_options = verify_correct_option(
                mt=mt,
                target=sample.ans_token_id,
                options=[
                    get_first_token_id(name=opt, tokenizer=mt.tokenizer, prefix=" ")
                    for opt in ["Yes", "No"]
                ],
                input=tokenized,
            )
            sample.metadata["tokenized"] = tokenized.data

            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                return get_counterfactual_samples_within_yes_no_task(
                    task=task,
                    mt=mt,
                    n_options=n_options,
                    clean_n_options=clean_n_options,
                    patch_n_options=patch_n_options,
                    prompt_template_idx=prompt_template_idx,
                    patch_prompt_template_idx=patch_prompt_template_idx,
                    clean_prompt_template_idx=clean_prompt_template_idx,
                    option_style=option_style,
                    patch_option_style=patch_option_style,
                    clean_option_style=clean_option_style,
                    filter_by_lm_prediction=filter_by_lm_prediction,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    verbose=verbose,
                    retry_count=retry_count + 1,
                )
            sample.prediction = [
                pred for token_id, (rank, pred) in track_options.items()
            ]

    # find the "?" token position in the samples
    for sample in [patch_sample, clean_sample]:
        tokenized = prepare_input(
            tokenizer=mt, prompts=sample.prompt(), return_offsets_mapping=True
        )
        offsets = tokenized.pop("offset_mapping")[0]
        ques_range = find_token_range(
            string=sample.prompt(),
            substring="?",
            tokenizer=mt.tokenizer,
            offset_mapping=offsets,
        )
        sample.metadata["ques_pos"] = ques_range[1] - 1
        sample.metadata["tokenized"] = tokenized.data

    return patch_sample, clean_sample


@torch.inference_mode()
def get_counterfactual_samples_within_first_task(
    task: SelectFirstTask | SelectLastTask,
    mt: ModelandTokenizer,
    patch_category: str | None = None,
    clean_category: str | None = None,
    n_options: int = 5,
    amt_to_sample: int = 2,
    prompt_template_idx=2,
    patch_prompt_template_idx: int | None = None,
    clean_prompt_template_idx: int | None = None,
    option_style="single_line",
    patch_option_style: str | None = None,
    clean_option_style: str | None = None,
    filter_by_lm_prediction: bool = True,
    distinct_options: bool = True,
    n_distractors: int = 5,
    retry_count: int = 0,
):
    # Set parameter defaults
    if patch_prompt_template_idx is None:
        patch_prompt_template_idx = prompt_template_idx
    if clean_prompt_template_idx is None:
        clean_prompt_template_idx = prompt_template_idx

    # Get the categories
    categories = list(task.category_wise_examples.keys())

    # Set the patch category
    if patch_category is None:
        patch_category = random.choice(categories)

    # Set the patch objects
    patch_objects = random.sample(
        task.category_wise_examples[patch_category], n_options
    )

    # Set the clean category
    if clean_category is None:
        clean_category = random.choice(
            list(
                set(categories)
                - {patch_category}
                - set(task.exclude_for_category(patch_category))
            )
        )

    # Set the clean objects
    clean_objects = random.sample(
        task.category_wise_examples[clean_category], n_options
    )

    # Set the other objects
    other_objects = []
    alt_other_objects = []
    other_categories = random.sample(
        list(
            set(categories)
            - (
                {patch_category, clean_category}
                | set(task.exclude_for_category(clean_category))
            )
        ),
        k=n_options,
    )
    ##print(f"{other_categories=}")
    for other_category in other_categories:
        other_examples = task.category_wise_examples[other_category]
        rand_other_example = random.choice(other_examples)
        other_objects.append(rand_other_example)
        alt_other_example = random.choice(other_examples)
        while rand_other_example == alt_other_example:
            alt_other_example = random.choice(other_examples)
        alt_other_objects.append(alt_other_example)
    # print(f"{other_objects=}")
    # print(f"{alt_other_objects=}")

    # Construct the clean and patch category options
    clean_category_options = random.sample(clean_objects, amt_to_sample)
    # print(f"{clean_category_options=}")
    patch_category_options = random.sample(patch_objects, amt_to_sample)
    # print(f"{patch_category_options=}")

    if distinct_options is not True:

        # Combine the clean and patch options
        combined_options = clean_category_options + patch_category_options

        # Add items form the other_objects list to pad it to n_options length
        for i in range(n_options - len(combined_options)):
            combined_options.append(other_objects.pop())

        clean_options = combined_options
        patch_options = combined_options

    else:
        # Get alternative patch options
        alt_patch_category_options = random.sample(
            [obj for obj in patch_objects if obj not in patch_category_options],
            amt_to_sample,
        )
        # Get alternative clean options
        alt_clean_category_options = random.sample(
            [obj for obj in clean_objects if obj not in clean_category_options],
            amt_to_sample,
        )
        # Compose prelimiary lists
        clean_options = clean_category_options + alt_patch_category_options
        patch_options = alt_clean_category_options + patch_category_options

        # Add the filler other objects
        for i in range(n_options - len(clean_options)):
            clean_options.append(other_objects.pop())
            patch_options.append(alt_other_objects.pop())

        # print(f"{clean_options=}")
        # print(f"{patch_options=}")

        # So I have these two lists and I need to make sure that their values line up by category.
        # So I start with two lists that definitely have their values line up: clean_category_options and patch_category_options
        # And these two lists' items align categorically as well with alt_clean_category_options and alt_patch_category_options respectively.
        # So now how to shuffle these in a random but synchronized way while adding more items to the list?
        # Probably makes sense to add the additional items to the list first.
        # That way I know everything is still in sync.
        # Ok, now I need some sort of randomized indexing process.
        # Maybe I can make a list of numbers to the range of the options
        # And then randomly swap these indices
        index_list = list(range(n_options))
        # print(f"{index_list=}")
        random.shuffle(index_list)
        # print(f"{index_list=}")

        clean_options = [clean_options[i] for i in index_list]
        patch_options = [patch_options[i] for i in index_list]

    patch_options_indices = []
    for option in patch_category_options:
        patch_options_indices.append(patch_options.index(option))
    patch_obj_idx = (
        min(patch_options_indices)
        if task.task_name == "select_first"
        else max(patch_options_indices)
    )

    # while True:
    #     # Gather the indices of the options of interest
    #     random.shuffle(clean_options)
    #     clean_options_indices = []
    #     for option in clean_category_options:
    #         clean_options_indices.append(clean_options.index(option))
    #     clean_first_idx = min(clean_options_indices)

    #     alt_patch_category_options_indices = []
    #     for option in alt_patch_category_options:
    #         alt_patch_category_options_indices.append(clean_options.index(option))
    #     alt_patch_first_idx = min(alt_patch_category_options_indices)
    #     if patch_first_idx != alt_patch_first_idx:
    #         break

    #############################################################
    # Shuffle clean_options once
    random.shuffle(clean_options)

    # Gather the indices of the options of interest
    clean_options_indices = []
    for option in clean_category_options:
        clean_options_indices.append(clean_options.index(option))
    clean_obj_idx = (
        min(clean_options_indices)
        if task.task_name == "select_first"
        else max(clean_options_indices)
    )

    alt_patch_category_options_indices = []
    for option in alt_patch_category_options:
        alt_patch_category_options_indices.append(clean_options.index(option))
    alt_patch_obj_idx = (
        min(alt_patch_category_options_indices)
        if task.task_name == "select_first"
        else max(alt_patch_category_options_indices)
    )

    # If indices are the same, swap to make them different
    if patch_obj_idx == alt_patch_obj_idx:
        # Find a safe swap position (not in clean_category_options or alt_patch_category_options)
        swap_idx = patch_obj_idx
        for i in range(n_options):
            if (
                i not in clean_options_indices
                and i not in alt_patch_category_options_indices
            ):
                swap_idx = i
                break

        # Swap the element at alt_patch_first_idx with the safe position
        clean_options[alt_patch_obj_idx], clean_options[swap_idx] = (
            clean_options[swap_idx],
            clean_options[alt_patch_obj_idx],
        )

        # Recalculate alt_patch_first_idx
        alt_patch_category_options_indices = []
        for option in alt_patch_category_options:
            alt_patch_category_options_indices.append(clean_options.index(option))
        alt_patch_obj_idx = (
            min(alt_patch_category_options_indices)
            if task.task_name == "select_first"
            else max(alt_patch_category_options_indices)
        )

    #############################################################

    # print(f"{patch_first_idx=} | {patch_options_indices=}")
    # print(f"{clean_first_idx=} | {clean_options_indices=}")

    # Store the information about the corresponding object from clean options
    clean_track_type_obj = clean_options[alt_patch_obj_idx]
    clean_track_first_token_id = get_first_token_id(
        clean_track_type_obj, mt.tokenizer, prefix=" "
    )

    clean_obj = clean_options[clean_obj_idx]

    clean_metadata = {
        "track_category": patch_category,
        "track_type_obj": clean_track_type_obj,
        "track_type_obj_idx": patch_obj_idx,
        "track_type_obj_token_id": clean_track_first_token_id,
    }
    patch_obj = patch_options[patch_obj_idx]

    patch_sample = SelectionSample(
        obj=patch_obj,
        answer=patch_obj,
        obj_idx=patch_obj_idx,
        ans_token_id=get_first_token_id(patch_obj, mt.tokenizer, prefix=" "),
        options=patch_options,
        category=patch_category,
        prompt_template=task.prompt_templates[patch_prompt_template_idx],
        default_option_style=patch_option_style or option_style,
    )

    clean_sample = SelectionSample(
        obj=clean_obj,
        answer=clean_obj,
        obj_idx=clean_obj_idx,
        ans_token_id=get_first_token_id(clean_obj, mt.tokenizer, prefix=" "),
        options=clean_options,
        category=clean_category,
        metadata=clean_metadata,
        prompt_template=task.prompt_templates[clean_prompt_template_idx],
        default_option_style=clean_option_style or option_style,
    )

    if filter_by_lm_prediction:
        test_samples = [patch_sample, clean_sample]
        if distinct_options is True:
            gold_sample = copy.deepcopy(patch_sample)
            gold_sample.options = clean_sample.options
            gold_sample.obj = clean_sample.metadata["track_type_obj"]
            gold_sample.obj_idx = clean_sample.metadata["track_type_obj_idx"]
            gold_sample.ans_token_id = clean_sample.metadata["track_type_obj_token_id"]
            test_samples.append(gold_sample)

        for sample in test_samples:
            if retry_count >= 10:
                break

            logger.debug(
                f"{sample.prompt()} >> {mt.tokenizer.decode(sample.ans_token_id)}"
            )
            tokenized = prepare_input(tokenizer=mt, prompts=sample.prompt())
            is_correct, predictions, track_options = verify_correct_option(
                mt=mt,
                target=sample.ans_token_id,
                options=sample.options,
                input=tokenized,
            )
            # print(f"{is_correct=}")
            # print(f"{predictions=}")
            # print(f"{track_options=}")

            sample.metadata["tokenized"] = tokenized.data
            sample.metadata["predictions"] = predictions

            if not is_correct:
                logger.error(
                    f'Prediction mismatch: {track_options[list(track_options.keys())[0]]}["{mt.tokenizer.decode(predictions[0].token_id)}"] != {sample.ans_token_id}["{mt.tokenizer.decode(sample.ans_token_id)}"]'
                )
                return get_counterfactual_samples_within_first_task(
                    task=task,
                    mt=mt,
                    patch_category=patch_category,
                    clean_category=clean_category,
                    n_options=n_options,
                    amt_to_sample=amt_to_sample,
                    prompt_template_idx=prompt_template_idx,
                    patch_prompt_template_idx=patch_prompt_template_idx,
                    clean_prompt_template_idx=clean_prompt_template_idx,
                    option_style=option_style,
                    patch_option_style=patch_option_style,
                    clean_option_style=clean_option_style,
                    filter_by_lm_prediction=True,
                    distinct_options=distinct_options,
                    n_distractors=n_distractors,
                    retry_count=retry_count + 1,
                )
            sample.prediction = predictions

    return patch_sample, clean_sample


get_counterfactual_samples_interface = {
    "select_one": get_counterfactual_samples_within_task,
    "select_order": get_counterfactual_samples_within_task,
    "counting": get_counterfactual_samples_within_counting_task,
    "yes_no": get_counterfactual_samples_within_yes_no_task,
    "select_first": get_counterfactual_samples_within_first_task,
    "select_last": get_counterfactual_samples_within_first_task,
}

import copy
import json
import logging
import os
import random
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from dataclasses_json import DataClassJsonMixin

from src.functional import detensorize, predict_next_token
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.selection.utils import KeyedSet, get_first_token_id, verify_correct_option
from src.tokens import prepare_input
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


@dataclass
class SelectionSample(DataClassJsonMixin):
    obj: str
    obj_idx: int
    prompt_template: str
    options: Sequence[str]
    answer: str | None = None  # if obj != answer
    subj: str | None = None
    category: str | None = None
    prediction: Optional[Sequence[PredictedToken]] = None
    ans_token_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line", "numbered", "bullet"] = "single_line"
    language: Literal["english", "spanish", "french"] = "english"

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
        assert (
            self.options[self.obj_idx] == self.obj
        ), "Object must be one of the options and match the object index."

    def __str__(self):
        return f"{self.subj} -> {self.obj} ({self.obj_idx}): {self.options}"

    def detensorize(self):
        self.metadata = detensorize(self.metadata)

    def prompt(
        self, option_style: Literal["single_line", "numbered"] | None = None
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
            if self.language == "english":
                options_str = f"Options: {options_str}."
            elif self.language == "spanish":
                options_str = f"Opciones: {options_str}."
            elif self.language == "french":
                options_str = f"Options : {options_str}."

        elif option_style == "numbered":
            options_str = "\n".join(
                f"{chr(ord('a') + i)}. {opt}" for i, opt in enumerate(self.options)
            )
        elif option_style == "bullet":
            options_str = "\n".join(f"* {opt}" for opt in self.options)
        else:
            raise ValueError(f"Invalid option_style: {option_style}.")

        prompt = prompt.replace("<_options_>", options_str)

        return prompt


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
    options: Sequence[str]
    count: int
    category: str | None = None
    prediction: Optional[Sequence[PredictedToken]] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line"] = "single_line"

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, Sequence):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")

    def __str__(self):
        return f"{self.count} {self.category} -> {self.options}: Ans: {self.prediction}"

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
    options: Sequence[str]
    yes_mode: bool = True
    category: str | None = None
    prediction: Optional[Sequence[PredictedToken]] = None
    metadata: dict = field(default_factory=dict)
    default_option_style: Literal["single_line"] = "single_line"

    def __post_init__(self):
        assert "<_options_>" in self.prompt_template
        if "<_category_>" in self.prompt_template:
            assert self.category is not None
        if not isinstance(self.options, Sequence):
            raise TypeError("Options must be a Sequence.")
        if len(self.options) < 2:
            raise ValueError("There must be at least two options.")

    def __str__(self):
        answer = "Yes" if self.yes_mode else "No"
        return f"{self.category} -> {self.options}: Ans: {answer}"
    
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


@dataclass
class SelectOneTask(DataClassJsonMixin):
    category_type: str
    prompt_templates: list[str]
    category_wise_examples: dict[str, list] = field(default_factory=dict)
    task_name: str = "select_one"

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
            return SelectOneTask(
                task_name="select_one",
                category_type=data.get("name", category_type),
                prompt_templates=data["prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
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
        subj = (
            random.choice(list(category_wise_examples[category].values))
            if subj is None
            else subj
        )

        # logger.debug(f"{subj=}")
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
            print(data)
            return SelectFirstTask(
                task_name="select_first",
                category_type=data.get("name", category_type),
                prompt_templates=data["first_item_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def __str__(self):
        return f"""SelectFirstTask: ({self.category_type})
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

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(DEFAULT_DATA_DIR, "counting/fruits.json"),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            counting_root = os.path.join(DEFAULT_DATA_DIR, "counting")
            path = os.path.join(counting_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return CountingTask(
                task_name="counting",
                category_type=data.get("name", category_type),
                prompt_templates=data["prompt_templates"],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def get_random_sample(
        self,
        mt: ModelandTokenizer,
        prompt_template_idx: int = 0,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        n_count: int = 2,
        n_distractors: int = 3,
        filter_by_lm_prediction: bool = True,
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

        print(f"Category: {category}")
        print(f"{category_wise_examples=}")

        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )

        category_keys = list(category_wise_examples.keys())
        assert (
            category_keys[0] == category
        ), "Main category must come first in the dataset categories"

        main_category_items = category_wise_examples[category_keys[0]].values
        counting_items = random.sample(main_category_items, k=n_count)

        generic_items = category_wise_examples[category_keys[1]].values
        distractors = random.sample(generic_items, k=n_distractors)

        if len(counting_items) > 1 and len(distractors) > 1:
            options = counting_items + distractors
        elif len(counting_items) == 1 and len(distractors) > 1:
            options = [counting_items] + distractors
        elif len(counting_items) > 1 and len(distractors) == 1:
            options = counting_items + [distractors]

        random.shuffle(options)

        sample = CountingSample(
            prompt_template=self.prompt_templates[prompt_template_idx],
            options=options,
            count=n_count,
            category=category,
            prediction=None,
            default_option_style=option_style,
        )

        if filter_by_lm_prediction:
            prompt = sample.prompt(option_style=option_style)
            inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata["tokenized"] = inputs.data

            predictions = predict_next_token(
                mt=mt,
                inputs=inputs,
            )[0]
            print(f"{predictions[0].token=}")
            count_str_map = {
                0: " zero",
                1: " one",
                2: " two",
                3: " three",
                4: " four",
                5: " five",
                6: " six",
                7: " seven",
                8: " eight",
                9: " nine",
                10: " ten",
            }
            if predictions[0].token != count_str_map[n_count]:
                logger.error(
                    f"""Sample = {sample}
                    Top prediction {predictions[0]} does not match the count '{n_count}'.
                    Retry count: {retry_count + 1}. Retrying ...
                    """
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    category=category,
                    n_count=n_count,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=True,
                    retry_count=retry_count + 1,
                )

            sample.prediction = predictions

        sample.metadata["retry_count"] = retry_count
        return sample

    def __str__(self):
        return f"""CountingTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""


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

    @staticmethod
    def load(
        path: PathLike | None = os.path.join(
            DEFAULT_DATA_DIR, "counting/fruits.json"
        ),
        category_type: str | None = None,
    ):
        if path is None:
            assert category_type is not None, "Path or category_type must be provided."
            counting_root = os.path.join(DEFAULT_DATA_DIR, "counting")
            path = os.path.join(counting_root, f"{category_type}.json")

        with open(path, "r") as f:
            data = json.load(f)
            return YesNoTask(
                task_name="yes_no",
                category_type=data.get("name", category_type),
                prompt_templates = [
                    "Are there any <_category_> in the following options.\nOptions: <_options_>\nAnswer Yes or No.\nAnswer:",
                    "<_options_>.\nHow many <_category_> are there in the previous options.\nAnswer Yes or No.\nAnswer:"
                ],
                category_wise_examples={k: v for k, v in data["categories"].items()},
            )

    def get_random_sample(
        self,
        mt: ModelandTokenizer = None,
        prompt_template_idx: int = 0,
        yes_mode: bool = True,
        option_style: Literal["single_line", "numbered"] = "single_line",
        category: str | None = None,
        n_distractors = 5,
        filter_by_lm_prediction: bool = False,
        retry_count: int = 0
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

        print(f"Category: {category}")
        print(f"{category_wise_examples=}")

        if category not in category_wise_examples:
            raise ValueError(
                f"Attribute '{category}' not found in {category_wise_examples.keys()}."
            )

        category_keys = list(category_wise_examples.keys())
        assert category_keys[0] == category, "Main category must come first in the dataset categories"

        main_category_items = category_wise_examples[category_keys[0]].values
        generic_items = category_wise_examples[category_keys[1]].values

        if yes_mode:
            yes_item = random.choice(main_category_items)
            distractors = random.sample(generic_items, k=n_distractors + 1)
        else:
            yes_item = None
            distractors = random.sample(generic_items, k=n_distractors + 1)

        if yes_item:
            options = [yes_item] + distractors
        else:
            options = distractors
        
        random.shuffle(options)

        sample = YesNoSample(
            prompt_template = self.prompt_templates[prompt_template_idx],
            options = options,
            yes_mode = yes_mode,
            category = category,
            prediction = None,
            default_option_style = option_style,
        )

        if filter_by_lm_prediction:
            if retry_count >= 100: return
            prompt = sample.prompt(option_style=option_style)
            inputs = prepare_input(prompts=prompt, tokenizer=mt)
            sample.metadata['tokenized'] = inputs.data
            correct_answer = "Yes" if yes_mode else "No"
            incorrect_answer = "No" if yes_mode else "Yes"
            correct_answer_token_id = get_first_token_id(correct_answer, tokenizer, prefix=" ")
            incorrect_answer_token_id = get_first_token_id(incorrect_answer, tokenizer, prefix=" ")

            is_correct, predictions, track_objs = verify_correct_option(
                mt=mt,
                input=inputs,
                target=correct_answer_token_id,
                options=[correct_answer_token_id, incorrect_answer_token_id],
                prefix=" "
            )
            # predictions = predict_next_token(
            #     mt=mt,
            #     inputs=inputs,
            # )[0]
            # 
            # print(f"{predictions[0].token=}")
            # answer = "Yes" if yes_mode else "No"
            # answer_token_id = get_first_token_id(answer, tokenizer, prefix=" ")
            # print(f"{answer_token_id=}")
            # print(f"{predictions[0].token_id=}")
            #if predictions[0].token_id != answer_token_id:
            if not is_correct:
                logger.error(
                    f"""Sample = {sample}
                    Top prediction {predictions[0]} does not match the correct answer '{correct_answer}'.
                    Retry count: {retry_count + 1}. Retrying ...
                    """
                )
                return self.get_random_sample(
                    mt=mt,
                    prompt_template_idx=prompt_template_idx,
                    option_style=option_style,
                    category=category,
                    n_distractors=n_distractors,
                    filter_by_lm_prediction=True,
                    retry_count=retry_count + 1,
                )

            sample.prediction = predictions 

        sample.metadata["retry_count"] = retry_count
        return sample   

    def __str__(self):
        return f"""YesNoTask: ({self.category_type})
Categories: {", ".join(f"{cat}({len(examples)})" for cat, examples in self.category_wise_examples.items())}
"""


# def load_people_by_category(
#     tokenizer: Tokenizer,
#     path: PathLike = os.path.join(DEFAULT_DATA_DIR, "selection_real/profession.json"),
#     category: str = None,
# ):

#     with open(path, "r") as f:
#         data = json.load(f)
#     people_by_category = {
#         k: KeyedSet(v, tokenizer=tokenizer) for k, v in data["categories"].items()
#     }
#     logger.info(f"Loaded {len(people_by_category)} categories")
#     return people_by_category


# def load_people_by_category_fakeverse(
#     tokenizer: Tokenizer,
#     path: PathLike = os.path.join(
#         DEFAULT_DATA_DIR, "synthetic_entities/64/profiles.json"
#     ),
#     category: str = "occupation",
# ):
#     """Load people by profession from a JSON file."""
#     with open(path, "r") as f:
#         data = json.load(f)

#     entity_by_category = defaultdict(list)
#     for entity in data:
#         entity_by_category[entity[category]].append(entity["name"])

#     people_by_category = {
#         k: KeyedSet(v, tokenizer=tokenizer) for k, v in entity_by_category.items()
#     }

#     logger.info(f"Loaded {len(people_by_category)} categories")
#     return people_by_category


# def get_random_sample(
#     people_by_category: dict[str, KeyedSet],
#     mt: ModelandTokenizer,
#     category: str = "occupation",
#     attribute: str | None = None,
#     subj: str | None = None,
#     n_distractors: int = 5,
#     filter_by_lm_prediction: bool = True,
#     obj_idx: int | None = None,
#     get_alt_obj: bool = False,  # TODO(arnab): Need to check accuracy with the alt obj as well
#     exclude_objs: Sequence[str] = [],
#     exclude_distractor_categories: Sequence[str] = [],
#     insert_distractor: Sequence[tuple[str, int]] = [],
#     retry_count: int = 0,
# ) -> SelectionSample:
#     """
#     Get a random sample with the specified attribute.
#     """

#     tokenizer = unwrap_tokenizer(mt)
#     if not people_by_category:
#         load_people_by_category_fakeverse(tokenizer, category=category)

#     attribute = attribute or random.choice(list(people_by_category.keys()))
#     if attribute not in people_by_category:
#         raise ValueError(
#             f"Attribute '{attribute}' not found in {people_by_category.keys()}."
#         )
#     kwargs = {
#         "subj": subj,
#         "obj_idx": obj_idx,
#     }
#     # print(f"Category: {category}")
#     # print(people_by_category[category].values)
#     subj = (
#         random.choice(list(people_by_category[attribute].values))
#         if subj is None
#         else subj
#     )
#     # logger.debug(f"{subj=}")
#     obj = random.choice(
#         (
#             people_by_category[attribute]
#             - KeyedSet([subj] + exclude_objs, tokenizer=tokenizer)
#         ).values
#     )
#     obj_token_id = get_first_token_id(obj, tokenizer, prefix=" ")
#     if obj_idx is None:
#         obj_idx = random.randint(0, n_distractors)
#     # logger.debug(f"{obj=}, {obj_token_id=}, {obj_idx=}, {exclude_objs=}")

#     obj_arr = [obj]
#     if get_alt_obj:
#         # Get an alternative object with the same attribute
#         alt_obj = random.choice(
#             (
#                 people_by_category[attribute]
#                 - KeyedSet([subj, obj] + exclude_objs, tokenizer=tokenizer)
#             ).values
#         )
#         obj_arr.append(alt_obj)
#     else:
#         alt_obj = None

#     distractors = []
#     obj_set = KeyedSet(obj_arr + exclude_objs, tokenizer=tokenizer)
#     other_attributes = random.sample(
#         list(
#             set(people_by_category.keys())
#             - set([attribute] + exclude_distractor_categories)
#         ),
#         k=n_distractors,
#     )
#     # print(other_categories)
#     for other_attribute in other_attributes:
#         distractors.append(
#             random.choice((people_by_category[other_attribute] - obj_set).values)
#         )

#     options = distractors[:obj_idx] + [obj] + distractors[obj_idx:]
#     for dist, idx in insert_distractor:
#         assert idx != obj_idx, "Cannot replace answer with a distractor."
#         assert idx < len(options), "Distractor index out of range."
#         options[idx] = dist
#     # logger.debug(f"{options=}")

#     metadata = {"attribute": attribute}
#     if get_alt_obj:
#         alt_obj_token_id = get_first_token_id(alt_obj, tokenizer, prefix=" ")
#         metadata["alt_obj"] = (alt_obj, alt_obj_token_id)
#     sample = SelectionSample(
#         match_with=subj,
#         obj=obj,
#         obj_idx=obj_idx,
#         options=options,
#         category=category,
#         obj_token_id=obj_token_id,
#         metadata=metadata,
#     )

#     if filter_by_lm_prediction:
#         prompt = sample.prompt
#         # logger.info(f"\nPrompt: {prompt}")
#         inputs = prepare_input(prompts=prompt, tokenizer=mt)
#         sample.metadata["tokenized"] = inputs.data

#         predictions = predict_next_token(
#             mt=mt,
#             inputs=inputs,
#         )[0]
#         if predictions[0].token_id != obj_token_id:
#             logger.error(
#                 f"""Sample = {sample}
# Top prediction {predictions[0]} does not match the object {obj}[{obj_token_id}, "{mt.tokenizer.decode(obj_token_id)}"].
# Retry count: {retry_count + 1}. Retrying ...
# """
#             )
#             return get_random_sample(
#                 people_by_category=people_by_category,
#                 mt=mt,
#                 n_distractors=n_distractors,
#                 get_alt_obj=get_alt_obj,
#                 filter_by_lm_prediction=filter_by_lm_prediction,
#                 category=category,
#                 attribute=attribute,
#                 exclude_objs=exclude_objs,
#                 exclude_distractor_categories=exclude_distractor_categories,
#                 insert_distractor=insert_distractor,
#                 retry_count=retry_count + 1,
#                 **kwargs,
#             )

#         sample.prediction = predictions

#     sample.metadata["retry_count"] = retry_count
#     return sample

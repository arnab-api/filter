import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"


@dataclass(frozen=False)
class InContextQuery(DataClassJsonMixin):
    """A query with a subject and object, and a context in which to embed it."""

    subject: str
    cf_description: str
    answer: str

    template: str = (
        "Assume an alternative universe where <subj> is in <loc>. In that universe, <subj> is located in the city of"
    )

    def set_template(self, template: str):
        self.template = template

    @property
    def query(self) -> str:
        return self.template.replace("<subj>", self.subject).replace(
            "<loc>", self.cf_description
        )

    def __str__(self) -> str:
        return f"{self.subject} -> {self.cf_description} | answer: {self.answer}"


@dataclass(frozen=True)
class RelationProperties(DataClassJsonMixin):
    """Some metadata about a relation."""

    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool
    # fn_type: str
    # disambiguating: bool


@dataclass(frozen=False)
class Relation(DataClassJsonMixin, Dataset):
    """An abstract mapping between subjects and objects.

    Attributes:
        name: The name of the relation, used as an ID.
        prompt_templates: Prompts representing the relation, where the subject is
            represented by {}.
        samples: A list of (subject, object) pairs satisfying the relation.
        properties: Relation metadata.
        _domain: Explicit list of all possible subjects. Accessed via the @property
            `domain`, which guesses the domain from the samples if not provided.
        _range: Equivalent to `_domain`, but for objects.
    """

    name: str
    prompt_templates: list[str]
    prompt_templates_zs: list[str]
    samples: list[RelationSample]
    properties: RelationProperties

    _prompt_template_idx: int = 0  # use the first prompt template by default
    _few_shot_samples: list[RelationSample] = field(default_factory=list)
    _few_shot_prefix: str | None = None
    _domain: list[str] | None = None
    _range: list[str] | None = None

    def __init__(
        self,
        name: str,
        prompt_templates: str,
        samples: list[RelationSample],
        properties: dict,
        prompt_templates_zs: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.prompt_templates = prompt_templates
        self.samples = samples
        self.properties = properties
        self._range = None
        self._range = self.range
        self.prompt_templates_zs = prompt_templates_zs

        self._few_shot_samples = []
        self.select_icl_examples(5)  # call the initialized object to change the number

        logger.info(f'initialized relation -> "{name}" with {len(self)} samples')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        query = self.prompt_templates[self._prompt_template_idx].format(
            self.samples[idx].subject
        )
        object = self.samples[idx].object
        full_query = self._few_shot_prefix + "\n" + query
        return (full_query, object)

    def select_icl_examples(self, num_icl):
        # Select few shot samples
        self._few_shot_samples = random.sample(
            self.samples + self._few_shot_samples, num_icl
        )
        self._few_shot_prefix = "\n".join(
            [
                self.prompt_templates[self._prompt_template_idx].format(demo.subject)
                + " "
                + demo.object
                for demo in self._few_shot_samples
            ]
        )
        self.samples = list(set(self.samples) - set(self._few_shot_samples))

    @property
    def domain(self) -> set[str]:
        if self._domain is not None:
            return set(self._domain)
        return {sample.subject for sample in self.samples}

    @property
    def range(self) -> set[str]:
        if self._range is not None:
            return set(self._range)
        return {sample.object for sample in self.samples}

    def without(self, sample: RelationSample) -> "Relation":
        """Return a copy of this relation without a given sample."""
        return self.set(samples=[s for s in self.samples if s != sample])

    def split(
        self, train_size: int, test_size: int | None = None
    ) -> tuple["Relation", "Relation"]:
        """Break into a train/test split."""
        if train_size > len(self.samples):
            raise ValueError(f"size must be <= {len(self.samples)}, got: {train_size}")
        if test_size is None:
            test_size = len(self.samples) - train_size

        # Shuffle once up front, because we're sometimes sorted, and if the relation
        # is 1:1, we'll always pick the same samples!
        samples = self.samples.copy()
        random.shuffle(samples)

        samples_by_object = defaultdict(list)
        for sample in samples:
            samples_by_object[sample.object].append(sample)

        for samples in samples_by_object.values():
            random.shuffle(samples)

        # List to store the result
        max_coverage_samples = []

        # As long as there are samples left
        while samples_by_object:
            # For each object
            for object in list(samples_by_object.keys()):
                # Add one sample to the result and remove it from the object's list
                max_coverage_samples.append(samples_by_object[object].pop(0))

                # If there are no more samples for this object, remove it from the dict
                if len(samples_by_object[object]) == 0:
                    del samples_by_object[object]

        train_samples = max_coverage_samples[:train_size]
        test_samples = max_coverage_samples[train_size : train_size + test_size]

        return (
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
        )

    def set(
        self,
        name: str | None = None,
        prompt_templates: Sequence[str] | None = None,
        prompt_templates_zs: Sequence[str] | None = None,
        properties: RelationProperties | None = None,
        samples: Sequence[RelationSample] | None = None,
        domain: Sequence[str] | None = None,
        range: Sequence[str] | None = None,
    ) -> "Relation":
        """Return a copy of this relation with any specified fields overwritten."""
        return Relation(
            name=name if name is not None else self.name,
            prompt_templates=(
                list(prompt_templates)
                if prompt_templates is not None
                else self.prompt_templates
            ),
            prompt_templates_zs=(
                list(prompt_templates_zs)
                if prompt_templates_zs is not None
                else self.prompt_templates_zs
            ),
            properties=properties if properties is not None else self.properties,
            samples=list(samples) if samples is not None else self.samples,
            _domain=list(domain) if domain is not None else self._domain,
            _range=list(range) if range is not None else self._range,
        )


class RelationDataset(Dataset[Relation]):
    """A torch dataset of relations."""

    def __init__(self, relations: list[Relation]):
        self.relations = relations

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> Relation:
        return self.relations[index]

    def filter(
        self,
        relation_names: Sequence[str] | None = None,
        **properties: bool | Sequence[str],
    ) -> "RelationDataset":
        relations = list(self.relations)
        if relation_names is not None:
            logger.debug(f"filtering to only relations: {relation_names}")
            relations = [r for r in relations if r.name in set(relation_names)]

        for key, value in properties.items():
            if value is not None:
                if isinstance(value, bool):
                    logger.debug(f"filtering by property {key}={value}")
                    matches = lambda x: x == value
                else:
                    logger.debug(f"filtering by property {key} in {value}")
                    value_set = set(value)
                    matches = lambda x: (x in value_set)

                relations = [
                    r for r in relations if matches(getattr(r.properties, key))
                ]

        return RelationDataset(relations)


def resolve_relation_file_path(file_name: str) -> Path:
    """Resolve the path to a relation file."""
    relation_path = os.path.join(DEFAULT_DATA_DIR, "relation")
    relation_categories = os.listdir(relation_path)
    for category in relation_categories:
        category_dir = Path(relation_path) / category
        if not category_dir.is_dir():
            continue
        for file in os.listdir(category_dir):
            if file == file_name:
                return Path(relation_path) / category / file
    raise FileNotFoundError(f"could not find relation file for {file_name}")


def load_relation_dict(file: PathLike, absolute_path=False) -> dict:
    """Load dict for a single relation from a json file."""
    file = resolve_relation_file_path(file) if not absolute_path else Path(file)
    if file.suffix != ".json":
        raise ValueError(f"relation files must be json, got: {file}")
    with file.open("r") as handle:
        relation_dict = json.load(handle)
    for key in ("domain", "range"):
        if key in relation_dict:
            relation_dict[f"_{key}"] = relation_dict.pop(key)

    # Check that all keys are valid kwargs to Relation
    valid_keys = set(field.name for field in fields(Relation))
    for key in relation_dict.keys():
        if key not in valid_keys:
            raise ValueError(
                f"invalid key in relation file {file}: {key}. "
                f"valid keys are: {valid_keys}"
            )

    # Compute the type of relation function (injection, surjection, bijection, etc.)
    # relation_dict["properties"]["fn_type"] = get_relation_fn_type(relation_dict)

    return relation_dict


def load_relation(file: PathLike, absolute_path=False) -> Relation:
    """Load a single relation from a json file."""
    relation_dict = load_relation_dict(file, absolute_path=absolute_path)
    return Relation(
        name=relation_dict["name"],
        prompt_templates=relation_dict["prompt_templates"],
        prompt_templates_zs=relation_dict["prompt_templates_zs"],
        samples=[
            RelationSample.from_dict(sample) for sample in relation_dict["samples"]
        ],
        properties=RelationProperties.from_dict(relation_dict["properties"]),
    )


def load_dataset(*paths: PathLike) -> RelationDataset:
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    if not paths:
        data_dir = DEFAULT_DATA_DIR
        logger.debug(f"no paths provided, using default data dir: {data_dir}")
        paths = (data_dir,)

    # Load all relation files
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            logger.debug(f"found relation file: {path}")
            files.append(path)
        else:
            logger.debug(f"{path} is directory, globbing for json files...")
            for file in sorted(path.glob("**/*.json")):
                logger.debug(f"found relation file: {file}")
                files.append(file)

    logger.debug(f"found {len(files)} relation files total, loading...")
    relation_dicts = [load_relation_dict(file) for file in files]

    # Mark all disambiguating relations
    domain_range_pairs: dict[tuple[str, str], int] = {}
    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        cur = domain_range_pairs.get((d, r), 0)
        domain_range_pairs[(d, r)] = cur + 1

    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        relation_dict["properties"]["disambiguating"] = domain_range_pairs[(d, r)] > 1

    # Create Relation objects
    relations = [Relation.from_dict(relation_dict) for relation_dict in relation_dicts]

    return RelationDataset(relations)

import random
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Mapping

# Test count weight is whatever is not consumed by training and validation
TRAINING_COUNT_WEIGHT: Final[float] = 0.8
VALIDATION_COUNT_WEIGHT: Final[float] = 0.1

PAD_TOKEN: Final[int] = 0
SEP_TOKEN: Final[int] = 257
SEQUENCE_LENGTH: Final[int] = 512
VOCABULARY_SIZE: Final[int] = 258

CLASS_NAMES: Final[tuple[str, ...]] = (
    "benign",
    "placeholder",
    "secret",
)

LABEL_TO_ID: Final[dict[str, int]] = {
    "benign": 0,
    "placeholder": 1,
    "secret": 2,
}

REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_id",
        "language",
        "path",
        "key",
        "value",
        "context",
        "label",
        "source",
    }
)


def split_repository_ids(
    repository_ids: Iterable[str], seed: int = 1337
) -> dict[str, list[str]]:
    unique_ids = sorted(set(repository_ids))

    random.Random(seed).shuffle(unique_ids)

    training_count = int(len(unique_ids) * TRAINING_COUNT_WEIGHT)
    validation_count = int(len(unique_ids) * VALIDATION_COUNT_WEIGHT)

    output: dict[str, list[str]] = {
        "train": unique_ids[:training_count],
        "validation": unique_ids[training_count : training_count + validation_count],
        "test": unique_ids[training_count + validation_count :],
    }

    return output


def take_prefix(data: bytes, budget: int) -> tuple[bytes, int]:
    amount = min(len(data), budget)
    return data[:amount], budget - amount


def closest_context_window(
    context: bytes,
    value: bytes,
    size: int,
) -> bytes:
    if size <= 0:
        return b""

    if len(context) <= size:
        return context

    match_start = context.find(value) if value else -1

    if match_start == -1:
        return context[:size]

    match_center = match_start + len(value) // 2
    window_start = match_center - size // 2
    window_start = max(0, min(window_start, len(context) - size))

    return context[window_start : window_start + size]


def encode_candidate(path: str, key: str, value: str, context: str) -> list[int]:
    encoded: list[int] = []

    budget: int = SEQUENCE_LENGTH - 3  # 3 separator tokens

    value_bytes, budget = take_prefix(value.encode("utf-8"), budget)
    key_bytes, budget = take_prefix(key.encode("utf-8"), budget)

    context_limit = min(128, budget)
    context_bytes = closest_context_window(
        context.encode("utf-8"),
        value.encode("utf-8"),
        context_limit,
    )
    budget -= len(context_bytes)

    path_data = path.encode("utf-8")
    path_amount = min(len(path_data), budget)
    path_bytes = path_data[-path_amount:] if path_amount else b""
    budget -= path_amount

    if budget > 0:
        budget += len(context_bytes)
        context_limit = budget
        context_bytes = closest_context_window(
            context.encode("utf-8"),
            value.encode("utf-8"),
            context_limit,
        )
        budget -= len(context_bytes)

    encoded.extend(byte + 1 for byte in path_bytes)
    encoded.append(SEP_TOKEN)

    encoded.extend(byte + 1 for byte in key_bytes)
    encoded.append(SEP_TOKEN)

    encoded.extend(byte + 1 for byte in value_bytes)
    encoded.append(SEP_TOKEN)

    encoded.extend(byte + 1 for byte in context_bytes)
    encoded.extend([PAD_TOKEN] * (SEQUENCE_LENGTH - len(encoded)))

    return encoded


def label_to_id(label: str) -> int:
    try:
        return LABEL_TO_ID[label]
    except KeyError as error:
        raise ValueError(f"Unknown label: {label}") from error


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number}")

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on line {line_number}: {', '.join(sorted(missing))}"
                )

            for field, value in record.items():
                if not isinstance(value, str):
                    raise ValueError(
                        f"Field {field!r} on line {line_number} must be a string"
                    )

            try:
                label_to_id(record["label"])
            except ValueError as error:
                raise ValueError(f"Invalid label on line {line_number}") from error

            records.append(record)

    if not records:
        raise ValueError("Dataset is empty")

    return records


def split_records(
    records: Iterable[Mapping[str, str]],
    seed: int = 1337,
) -> dict[str, list[Mapping[str, str]]]:
    records_list = list(records)
    repository_ids = [record["repository_id"] for record in records_list]

    split_repo_ids = split_repository_ids(repository_ids, seed)

    repository_to_split = {}

    for split_name, repository_ids in split_repo_ids.items():
        for repository_id in repository_ids:
            repository_to_split[repository_id] = split_name

    split_record_lists: dict[str, list[Mapping[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for record in records_list:
        repository_id = record["repository_id"]
        split_name = repository_to_split[repository_id]
        split_record_lists[split_name].append(record)

    return split_record_lists

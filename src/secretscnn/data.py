import random
from collections.abc import Iterable
from typing import Final

# Test count weight is whatever is not consumed by training and validation
TRAINING_COUNT_WEIGHT: Final[float] = 0.8
VALIDATION_COUNT_WEIGHT: Final[float] = 0.1

PAD_TOKEN: Final[int] = 0
SEP_TOKEN: Final[int] = 257
SEQUENCE_LENGTH: Final[int] = 512
VOCABULARY_SIZE: Final[int] = 258


def split_repository_ids(
    repository_ids: Iterable[str], seed: int = 1337
) -> dict[str, list[str]]:
    unique_ids = sorted(set(repository_ids))

    random.Random(seed).shuffle(unique_ids)

    training_count = int(len(unique_ids) * TRAINING_COUNT_WEIGHT)
    validation_count = int(len(unique_ids) * VALIDATION_COUNT_WEIGHT)

    output = {
        "train" : unique_ids[:training_count],
        "validation": unique_ids[training_count:training_count+validation_count],
        "test": unique_ids[training_count+validation_count:]
    }

    return output


def encode_candidate(path: str, key: str, value: str, context: str) -> list[int]:

    encoded: list[int] = []

    encoded.extend([byte + 1 for byte in path.encode("utf-8")])
    encoded.append(SEP_TOKEN)

    encoded.extend([byte + 1 for byte in key.encode("utf-8")])
    encoded.append(SEP_TOKEN)

    encoded.extend([byte + 1 for byte in value.encode("utf-8")])
    encoded.append(SEP_TOKEN)

    encoded.extend([byte + 1 for byte in context.encode("utf-8")])

    if (len(encoded) > SEQUENCE_LENGTH):
        raise ValueError(f"More than SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) tokens in encoded list")

    while (len(encoded) < SEQUENCE_LENGTH):
        encoded.append(PAD_TOKEN)

    return encoded


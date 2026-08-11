import random

# Test count weight is whatever is not consumed by training and validation
TRAINING_COUNT_WEIGHT = 0.8
VALIDATION_COUNT_WEIGHT = 0.1


def split_repository_ids(repository_ids, seed=1337):
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

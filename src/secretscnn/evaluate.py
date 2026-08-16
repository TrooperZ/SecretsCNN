import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import TypedDict
from secretscnn.data import CLASS_NAMES, LABEL_TO_ID
import re

EXACT_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
)

PLACEHOLDER_VALUES: frozenset[str] = frozenset({
    "your-token-here",
    "your_api_key",
    "replace-me",
    "replace_with_token",
    "changeme",
    "placeholder",
    "example-secret",
    "test-token",
    "redacted",
    "xxxxxxxx",
    "fakekey123",
})

REFERENCE_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}"),
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*"),
)

class ClassMetrics(TypedDict):
    precision: float
    recall: float
    f1: float


class EvaluationReport(TypedDict):
    confusion_matrix: list[list[int]]
    per_class: dict[str, ClassMetrics]
    macro_f1: float
    benign_false_positives_per_1000: float
    placeholder_predicted_secret: int
    secret_predicted_placeholder: int


def calculate_class_metrics(
    matrix: list[list[int]],
    class_id: int,
) -> tuple[float, float, float]:
    true_positive  = matrix[class_id][class_id]
    false_negative = sum(matrix[class_id]) - true_positive
    false_positive = sum(row[class_id] for row in matrix) - true_positive

    precision_total = true_positive + false_positive
    precision = true_positive / precision_total if precision_total else 0.0

    recall_total = true_positive + false_negative
    recall = true_positive / recall_total if recall_total else 0.0

    f1_total = precision + recall
    f1 = 2 * precision * recall / f1_total if f1_total else 0.0

    return (precision, recall, f1)


def build_confusion_matrix(
    actual_labels: list[int],
    predicted_labels: list[int],
    class_count: int = 3,
) -> list[list[int]]:

    if len(actual_labels) != len(predicted_labels):
        raise ValueError("actual and predicted labels must have equal lengths")

    confusion_matrix = [
        [0 for _ in range(class_count)]
        for _ in range(class_count)
    ]

    for actual, predicted in zip(actual_labels, predicted_labels):
        confusion_matrix[actual][predicted] += 1

    return confusion_matrix

def calculate_macro_f1(matrix: list[list[int]]) -> float:
    if not matrix:
        return 0.0

    f1_scores = [
        calculate_class_metrics(matrix, class_id)[2]
        for class_id in range(len(matrix))
    ]
    return sum(f1_scores) / len(f1_scores)

def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
) -> tuple[list[int], list[int]]:
    model.eval()
    actual_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    return actual_labels, predicted_labels

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
) -> EvaluationReport:
    actual, predicted = collect_predictions(model, data_loader)
    return evaluate_predictions(actual, predicted)


def calculate_benign_false_positives_per_1000(
    matrix: list[list[int]],
) -> float:
    benign_id = LABEL_TO_ID["benign"]
    secret_id = LABEL_TO_ID["secret"]

    benign_total = sum(matrix[benign_id])
    if benign_total == 0:
        return 0.0

    false_positives = matrix[benign_id][secret_id]
    return false_positives / benign_total * 1000

def calculate_placeholder_secret_confusion(
    matrix: list[list[int]],
) -> tuple[int, int]:
    placeholder_id = LABEL_TO_ID["placeholder"]
    secret_id = LABEL_TO_ID["secret"]

    return (
        matrix[placeholder_id][secret_id],
        matrix[secret_id][placeholder_id],
    )

def matches_exact_secret_signature(value: str) -> bool:
    stripped_value = value.strip()
    return any(
        pattern.fullmatch(stripped_value) is not None
        for pattern in EXACT_SECRET_PATTERNS
    )

def matches_literal_placeholder(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_VALUES

def matches_reference_placeholder(value: str) -> bool:
    stripped_value = value.strip()
    return any(
        pattern.fullmatch(stripped_value) is not None
        for pattern in REFERENCE_PLACEHOLDER_PATTERNS
    )

def predict_regex_baseline(value: str) -> int:
    if matches_exact_secret_signature(value):
        return LABEL_TO_ID["secret"]

    if matches_literal_placeholder(value) or matches_reference_placeholder(value):
        return LABEL_TO_ID["placeholder"]

    return LABEL_TO_ID["benign"]

def evaluate_predictions(
    actual_labels: list[int],
    predicted_labels: list[int],
) -> EvaluationReport:
    matrix = build_confusion_matrix(
        actual_labels,
        predicted_labels,
        class_count=len(CLASS_NAMES),
    )

    per_class: dict[str, ClassMetrics] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        precision, recall, f1 = calculate_class_metrics(matrix, class_id)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    placeholder_predicted_secret, secret_predicted_placeholder = (
        calculate_placeholder_secret_confusion(matrix)
    )

    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1": calculate_macro_f1(matrix),
        "benign_false_positives_per_1000": (
            calculate_benign_false_positives_per_1000(matrix)
        ),
        "placeholder_predicted_secret": placeholder_predicted_secret,
        "secret_predicted_placeholder": secret_predicted_placeholder,
    }
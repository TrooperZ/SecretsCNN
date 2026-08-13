import torch
from torch import nn
from collections.abc import Iterable, Mapping

from secretscnn.data import encode_candidate, label_to_id


def calculate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    # loss = -log(correct-class probability)
    return nn.functional.cross_entropy(logits, labels)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = calculate_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def records_to_tensors(
    records: Iterable[Mapping[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded_records: list[list[int]] = []
    label_ids: list[int] = []

    for record in records:
        encoded_records.append(
            encode_candidate(
                path=record["path"],
                key=record["key"],
                value=record["value"],
                context=record["context"],
            )
        )
        label_ids.append(label_to_id(record["label"]))
    return (
        torch.tensor(encoded_records, dtype=torch.long),
        torch.tensor(label_ids, dtype=torch.long),
    )

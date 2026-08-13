import torch
from torch import nn
from collections.abc import Iterable, Mapping
from pathlib import Path
from secretscnn.data import encode_candidate, label_to_id
from torch.utils.data import DataLoader, TensorDataset
from typing import TypedDict

class EpochResult(TypedDict):
    epoch: int
    training_loss: float
    validation_loss: float

def calculate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    # Note that loss = -log(correct-class probability)
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

    if not encoded_records:
        raise ValueError("Cannot create tensors from an empty record collection")
    return (
        torch.tensor(encoded_records, dtype=torch.long),
        torch.tensor(label_ids, dtype=torch.long),
    )


def create_data_loader(
    records: Iterable[Mapping[str, str]],
    batch_size: int,
    shuffle: bool,
    seed: int = 1337,
) -> DataLoader:
    input_ids, labels = records_to_tensors(records)
    dataset = TensorDataset(input_ids, labels)
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for input_ids, labels in data_loader:
        batch_loss = train_step(model, optimizer, input_ids, labels)
        batch_size = labels.shape[0]

        total_loss += batch_loss * batch_size
        total_examples += batch_size

    return total_loss / total_examples

def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
) -> float:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            batch_loss = calculate_loss(logits, labels).item()
            batch_size = labels.shape[0]

            total_loss += batch_loss * batch_size
            total_examples += batch_size

    return total_loss / total_examples

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    checkpoint_path: str | Path,
) -> list[EpochResult]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_validation_loss = float("inf")
    history = []

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(model, optimizer, train_loader)
        validation_loss = evaluate_loss(model, validation_loader)

        history.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint,
            )

    return history
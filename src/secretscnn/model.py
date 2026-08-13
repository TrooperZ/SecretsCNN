import torch
from torch import nn

from secretscnn.data import PAD_TOKEN, VOCABULARY_SIZE

EMBEDDING_DIM = 16
CONV_FILTERS = 32
KERNEL_WIDTHS = (3, 5, 7)
HIDDEN_DIM = 32
CLASS_COUNT = 3

class ByteEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PAD_TOKEN
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.embedding_layer(input_ids)
        assert output.shape == (*input_ids.shape, EMBEDDING_DIM)
        return output


class ConvBranch(nn.Module):
    def __init__(self, kernel_width: int):
        super().__init__()
        self.kernel_width = kernel_width
        self.conv = nn.Conv1d(
            in_channels=EMBEDDING_DIM,
            out_channels=CONV_FILTERS,
            kernel_size=kernel_width,
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        channels_first = embedded.transpose(1, 2)
        activations = self.conv(channels_first)
        output_length = embedded.shape[1] - self.kernel_width + 1
        activated = torch.relu(activations)
        pooled = torch.amax(activated, dim=2)

        assert activations.shape == (
            embedded.shape[0],
            CONV_FILTERS,
            output_length,
        )

        assert pooled.shape == (embedded.shape[0], CONV_FILTERS)

        return pooled

class SecretsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ByteEmbedding()

        self.branches = nn.ModuleList(
            ConvBranch(width) for width in KERNEL_WIDTHS
        )

        self.hidden = nn.Linear(
            CONV_FILTERS * len(KERNEL_WIDTHS),
            HIDDEN_DIM,
        )

        self.output = nn.Linear(HIDDEN_DIM, CLASS_COUNT)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        pooled_branches = [branch(embedded) for branch in self.branches]
        features = torch.cat(pooled_branches, dim=1)
        hidden = torch.relu(self.hidden(features))
        logits = self.output(hidden)

        assert logits.shape == (input_ids.shape[0], CLASS_COUNT)
        return logits
import torch
from torch import nn

from secretscnn.data import PAD_TOKEN, VOCABULARY_SIZE

EMBEDDING_DIM = 16

class ByteEmbedding(nn.Module):

    def __init__(self):
        nn.Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PAD_TOKEN
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass
import unittest

import torch

from secretscnn.data import PAD_TOKEN, SEQUENCE_LENGTH
from secretscnn.model import CLASS_COUNT, SecretsCNN


class SecretsCNNTest(unittest.TestCase):
    def test_forward_contract(self):
        model = SecretsCNN()
        input_ids = torch.zeros((2, SEQUENCE_LENGTH), dtype=torch.long)

        logits = model(input_ids)

        self.assertEqual(logits.shape, (2, CLASS_COUNT))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            500_000,
        )
        self.assertEqual(
            torch.count_nonzero(
                model.embedding.embedding_layer.weight[PAD_TOKEN]
            ).item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()

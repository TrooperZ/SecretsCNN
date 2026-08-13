import tempfile
import unittest
from pathlib import Path

import torch

from secretscnn.model import SecretsCNN
from secretscnn.train import create_data_loader, train_model


class TrainingTest(unittest.TestCase):
    def test_training_saves_best_checkpoint(self):
        records = [
            {"path": "a.py", "key": "name", "value": "value", "context": "name=value", "label": "benign"},
            {"path": "b.env", "key": "TOKEN", "value": "replace-me", "context": "TOKEN=replace-me", "label": "placeholder"},
            {"path": "c.toml", "key": "password", "value": "synthetic_pw_42", "context": "password=synthetic_pw_42", "label": "secret"},
        ]

        torch.manual_seed(7)
        model = SecretsCNN()
        train_loader = create_data_loader(records * 2, 3, shuffle=True, seed=7)
        validation_loader = create_data_loader(records, 2, shuffle=False)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "models" / "secretcnn.pt"
            history = train_model(
                model,
                train_loader,
                validation_loader,
                epochs=2,
                checkpoint_path=checkpoint_path,
            )

            self.assertEqual([result["epoch"] for result in history], [1, 2])
            self.assertTrue(checkpoint_path.is_file())

            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(
                checkpoint["validation_loss"],
                min(result["validation_loss"] for result in history),
            )

            restored = SecretsCNN()
            restored.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    unittest.main()

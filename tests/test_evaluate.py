import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from secretscnn.evaluate import (
    build_confusion_matrix,
    calculate_benign_false_positives_per_1000,
    calculate_class_metrics,
    calculate_macro_f1,
    calculate_placeholder_secret_confusion,
    evaluate_model,
)


class EvaluationTest(unittest.TestCase):
    def test_metrics_from_predictions(self):
        actual = [0] * 100 + [1] * 100 + [2] * 100
        predicted = (
            [0] * 90 + [1] * 5 + [2] * 5
            + [1] * 85 + [2] * 15
            + [1] * 4 + [2] * 96
        )

        matrix = build_confusion_matrix(actual, predicted)

        self.assertEqual(matrix, [[90, 5, 5], [0, 85, 15], [0, 4, 96]])
        self.assertEqual(calculate_placeholder_secret_confusion(matrix), (15, 4))
        self.assertEqual(calculate_benign_false_positives_per_1000(matrix), 50.0)
        self.assertAlmostEqual(calculate_class_metrics(matrix, 2)[0], 96 / 116)
        self.assertAlmostEqual(calculate_class_metrics(matrix, 2)[1], 0.96)
        self.assertGreater(calculate_macro_f1(matrix), 0.89)

    def test_evaluate_model_returns_structured_report(self):
        logits = torch.tensor([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ])
        labels = torch.tensor([0, 1, 2, 1])
        data_loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

        report = evaluate_model(nn.Identity(), data_loader)

        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertEqual(report["placeholder_predicted_secret"], 1)
        self.assertEqual(report["secret_predicted_placeholder"], 0)
        self.assertAlmostEqual(report["macro_f1"], 7 / 9)


if __name__ == "__main__":
    unittest.main()

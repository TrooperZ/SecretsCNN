import random
import unittest

from secretscnn.data import (
    PAD_TOKEN,
    SEP_TOKEN,
    SEQUENCE_LENGTH,
    encode_candidate,
    split_repository_ids,
)


class SplitRepositoryIdsTest(unittest.TestCase):
    def test_split_is_grouped_and_deterministic(self):
        repository_ids = [f"repo-{index}" for index in range(20)] + ["repo-3", "repo-7"]

        first = split_repository_ids(repository_ids, seed=7)
        second = split_repository_ids(reversed(repository_ids), seed=7)
        assigned = first["train"] + first["validation"] + first["test"]

        self.assertEqual(first, second)
        self.assertEqual(set(assigned), set(repository_ids))
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [16, 2, 2])

        random.seed(99)
        expected_next_random = random.random()
        random.seed(99)
        split_repository_ids(repository_ids, seed=7)
        self.assertEqual(random.random(), expected_next_random)

    def test_encode_candidate(self):
        tokens = encode_candidate("a.py", "K", "x", "")

        self.assertEqual(len(tokens), SEQUENCE_LENGTH)
        self.assertEqual(
            tokens[:12],
            [98, 47, 113, 122, SEP_TOKEN, 76, SEP_TOKEN, 121, SEP_TOKEN, 0, 0, 0],
        )
        self.assertTrue(all(token == PAD_TOKEN for token in tokens[9:]))

        tokens = encode_candidate("é", "", "", "")

        self.assertEqual(
            tokens[:5],
            [196, 170, SEP_TOKEN, SEP_TOKEN, SEP_TOKEN],
        )

        tokens = encode_candidate("", "", "", "")

        self.assertEqual(tokens[:3], [SEP_TOKEN, SEP_TOKEN, SEP_TOKEN])
        self.assertTrue(all(token == PAD_TOKEN for token in tokens[3:]))

        tokens = encode_candidate("ignored.py", "API_TOKEN", "x" * 700, "ignored")

        self.assertEqual(tokens[:3], [SEP_TOKEN, SEP_TOKEN, ord("x") + 1])
        self.assertEqual(tokens.count(ord("x") + 1), SEQUENCE_LENGTH - 3)
        self.assertEqual(tokens[-1], SEP_TOKEN)

        value = "synthetic_secret_123"
        context = "a" * 300 + f"API_TOKEN={value}" + "b" * 300
        tokens = encode_candidate(
            "discard/" * 100 + "src/config.py",
            "API_TOKEN",
            value,
            context,
        )
        first_sep = tokens.index(SEP_TOKEN)
        second_sep = tokens.index(SEP_TOKEN, first_sep + 1)
        third_sep = tokens.index(SEP_TOKEN, second_sep + 1)

        kept_path = bytes(token - 1 for token in tokens[:first_sep])
        kept_key = bytes(token - 1 for token in tokens[first_sep + 1:second_sep])
        kept_value = bytes(token - 1 for token in tokens[second_sep + 1:third_sep])
        kept_context = bytes(
            token - 1 for token in tokens[third_sep + 1:] if token != PAD_TOKEN
        )

        self.assertTrue(kept_path.endswith(b"src/config.py"))
        self.assertEqual(kept_key, b"API_TOKEN")
        self.assertEqual(kept_value, value.encode("utf-8"))
        self.assertIn(value.encode("utf-8"), kept_context)


if __name__ == "__main__":
    unittest.main()

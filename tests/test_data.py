import random
import unittest

from secretscnn.data import split_repository_ids


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


if __name__ == "__main__":
    unittest.main()

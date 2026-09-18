import unittest

import torch

from experiments.run_mechanism_checks import check_bfs, reference_bfs
from finite_cot.models import boolean_bfs


class BooleanBFSTests(unittest.TestCase):
    def test_reported_random_graph_grid(self):
        result = check_bfs()

        self.assertEqual(result["cases"], 81)
        self.assertEqual(result["exact_frontier_agreements"], 81)
        self.assertEqual(result["decision_accuracy"], 1.0)
        self.assertEqual(result["frontier_vector_agreement"], 1.0)
        self.assertEqual(result["failures"], [])
        self.assertTrue(result["passed"])

    def test_directed_path_activates_one_hop_per_update(self):
        # 0 -> 1 -> 2 -> 3, plus an isolated vertex. The reverse edges are
        # absent, which also detects an adjacency-orientation error.
        adjacency = torch.zeros(5, 5, dtype=torch.bool)
        adjacency[0, 1] = True
        adjacency[1, 2] = True
        adjacency[2, 3] = True

        for steps in range(5):
            with self.subTest(steps=steps):
                actual = boolean_bfs(
                    adjacency.unsqueeze(0), torch.tensor([0]), steps
                )[0]
                expected = torch.tensor(
                    [vertex <= min(steps, 3) for vertex in range(4)] + [False]
                )
                self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_reached_vertices_persist_through_cycle(self):
        adjacency = torch.zeros(4, 4, dtype=torch.bool)
        adjacency[0, 1] = True
        adjacency[1, 0] = True
        adjacency[1, 2] = True

        actual = boolean_bfs(adjacency.unsqueeze(0), torch.tensor([0]), 4)[0]
        self.assertTrue(torch.equal(actual.cpu(), torch.tensor([1, 1, 1, 0]).bool()))

    def test_batched_graphs_and_sources(self):
        first = torch.zeros(4, 4, dtype=torch.bool)
        first[0, 1] = first[1, 2] = True
        second = torch.zeros(4, 4, dtype=torch.bool)
        second[3, 2] = second[2, 1] = True
        adjacency = torch.stack((first, second))
        sources = torch.tensor([0, 3])

        actual = boolean_bfs(adjacency, sources, steps=1).cpu()
        expected = torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_reference_is_limited_by_shortest_path_depth(self):
        adjacency = torch.zeros(4, 4, dtype=torch.bool)
        adjacency[0, 1] = adjacency[1, 2] = adjacency[2, 3] = True

        self.assertTrue(
            torch.equal(
                reference_bfs(adjacency, source=0, steps=2),
                torch.tensor([True, True, True, False]),
            )
        )


if __name__ == "__main__":
    unittest.main()

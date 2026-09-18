import unittest

import torch

from experiments.run_mechanism_checks import check_rare_witness
from finite_cot.rare_witness import (
    decode_hypercube,
    exhaustive_scan,
    fixed_point_round,
    hypercube_codes,
    sampled_inspection,
    sampled_inspection_prefixes,
)
from finite_cot.theory import required_inspections, sampling_success


class RareWitnessPrimitiveTests(unittest.TestCase):
    def test_codes_are_msb_first_and_decode_to_identity(self):
        codes = hypercube_codes(3)
        expected = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(codes, expected))
        self.assertTrue(torch.equal(decode_hypercube(codes), torch.arange(8)))

    def test_sampling_budgets_share_one_prefix_stream(self):
        markers = torch.eye(8)
        budgets = (0, 1, 2, 4, 8)
        curve = sampled_inspection_prefixes(
            markers, budgets, generator=torch.Generator().manual_seed(23)
        )
        final_only = sampled_inspection(
            markers, inspections=8, generator=torch.Generator().manual_seed(23)
        )

        self.assertTrue(torch.equal(curve.success[-1], final_only.success))
        self.assertTrue(bool((curve.success[1:] >= curve.success[:-1]).all()))
        self.assertFalse(bool(curve.success[0].any()))
        self.assertEqual(curve.inspections, 8)

    def test_exhaustive_scan_handles_null_and_rejects_multiple_witnesses(self):
        markers = torch.zeros(3, 4)
        markers[0, 1] = 1
        markers[1, 3] = 1
        scanned = exhaustive_scan(markers)
        self.assertTrue(torch.equal(scanned.prediction, torch.tensor([1, 3, 4])))
        self.assertEqual(scanned.marker_accesses, 4)

        markers[2, :2] = 1
        with self.assertRaisesRegex(ValueError, "at most one witness"):
            exhaustive_scan(markers)

    def test_bounded_noise_preserves_marked_and_null_decoding(self):
        codes = hypercube_codes(4)
        adversarial_noise = -0.499 * codes
        self.assertTrue(
            torch.equal(decode_hypercube(codes + adversarial_noise), torch.arange(16))
        )
        null_noise = torch.tensor([[0.499, -0.499, 0.499, -0.499]])
        self.assertEqual(decode_hypercube(null_noise).item(), 16)

    def test_fixed_point_midpoints_use_ties_to_even(self):
        self.assertTrue(
            torch.equal(
                fixed_point_round(torch.tensor([-0.25, 0.25]), fractional_bits=1),
                torch.zeros(2),
            )
        )

    def test_sampling_formula_and_two_thirds_threshold(self):
        branches = 64
        threshold = required_inspections(1 / branches, delta=1 / 3)
        self.assertGreaterEqual(sampling_success(1 / branches, threshold), 2 / 3)
        self.assertLess(sampling_success(1 / branches, threshold - 1), 2 / 3)


class RareWitnessProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = check_rare_witness()

    def test_paired_sampling_grid_and_monte_carlo_tolerance(self):
        self.assertEqual(self.result["sampling_dimensions"], [4, 6, 8, 10])
        self.assertEqual(self.result["trials_per_sampling_dimension"], 5000)
        self.assertTrue(self.result["paired_sampling_prefixes"])
        self.assertEqual(len(self.result["sampling"]), 28)
        for branches in (16, 64, 256, 1024):
            rows = [
                row for row in self.result["sampling"] if row["branches"] == branches
            ]
            self.assertEqual(
                [row["inspections"] for row in rows],
                [
                    0,
                    1,
                    branches // 4,
                    branches // 2,
                    branches,
                    2 * branches,
                    4 * branches,
                ],
            )
            for row in rows:
                self.assertLessEqual(row["ci95_low"], row["empirical_success"])
                self.assertGreaterEqual(row["ci95_high"], row["empirical_success"])
                self.assertEqual(
                    row["total_branch_reads"],
                    row["branch_reads_per_instance"] * row["trials"],
                )
        summary = self.result["summary"]
        self.assertLess(summary["maximum_absolute_error"], 0.04)
        self.assertLess(summary["maximum_standardized_error"], 4.0)
        self.assertTrue(summary["sampling_passed"])

    def test_all_witness_positions_and_null_cases_are_exact(self):
        self.assertEqual(self.result["construction_dimensions"], [4, 6, 8, 10, 12])
        for row in self.result["construction"]:
            self.assertEqual(row["positions_tested"], row["branches"])
            self.assertEqual(row["aggregation_accuracy"], 1.0)
            self.assertEqual(row["exhaustive_scan_accuracy"], 1.0)
            self.assertEqual(row["bounded_noise_accuracy"], 1.0)
            self.assertTrue(row["aggregation_null_correct"])
            self.assertTrue(row["scan_null_correct"])
            self.assertTrue(row["null_noise_correct"])
            self.assertEqual(row["aggregation_branch_reads"], row["branches"])
            self.assertEqual(
                row["aggregation_scalar_work"], row["branches"] * row["dimension"]
            )
        self.assertTrue(self.result["summary"]["construction_passed"])

    def test_precision_transition_matches_fixed_point_prediction(self):
        for row in self.result["precision"]:
            expected = (
                1.0
                if not row["normalized"] or row["precision_offset"] >= 0
                else 0.0
            )
            self.assertEqual(row["accuracy"], expected)
        self.assertTrue(self.result["summary"]["precision_passed"])
        self.assertTrue(self.result["passed"])


if __name__ == "__main__":
    unittest.main()

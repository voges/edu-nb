"""Tests for the alignment module."""


import unittest
import numpy as np
import alignment as aln


class AlignTestCase(unittest.TestCase):
    """Test case for the align function."""

    def test_gattaca(self):
        """Test a random alignment."""
        seq_a = ["G", "A", "T", "T", "A", "C", "A"]
        seq_b = ["C", "A", "T", "A", "C", "C", "C", "A"]
        align = aln.make_align(mode="global")

        seq_a_aln, seq_b_aln, warp_path_a, warp_path_b, scoring_mat = align(
            seq_a=seq_a, seq_b=seq_b
        )
        self.assertEqual(seq_a_aln, ["G", "A", "T", "T", "A", "-", "-", "C", "A"])
        self.assertEqual(seq_b_aln, ["C", "A", "-", "T", "A", "C", "C", "C", "A"])
        self.assertEqual(warp_path_a, [0, 1, 2, 3, 4, 4, 4, 5, 6])
        self.assertEqual(warp_path_b, [0, 1, 1, 2, 3, 4, 5, 6, 7])
        scoring_mat_gt = np.array(
            [
                [-1, -2, -3, -4, -5, -6, -7, -8],
                [-2, 0, -1, -2, -3, -4, -5, -6],
                [-3, -1, 1, 0, -1, -2, -3, -4],
                [
                    -4,
                    -2,
                    0,
                    0,
                    -1,
                    -2,
                    -3,
                    -4,
                ],
                [-5, -3, -1, 1, 0, -1, -2, -2],
                [-4, -4, -2, 0, 2, 1, 0, -1],
                [-5, -3, -3, -1, 1, 1, 0, 1],
            ]
        )
        self.assertEqual(scoring_mat.all(), scoring_mat_gt.all())

    def test_int_list_input(self):
        """Test lists with integers as input."""
        seq_a = [0, 0, 1, 0, 0]
        seq_b = [0, 1, 1, 0]
        align = aln.make_align(mode="global")
        seq_a_aln, seq_b_aln, _, _, _ = align(seq_a=seq_a, seq_b=seq_b)
        self.assertEqual(seq_a_aln, [0, "-", 0, 1, 0, 0])
        self.assertEqual(seq_b_aln, [1, 0, 0, 1, 1, 0])

    def test_string_input(self):
        """Test strings as input."""
        seq_a = "GATTACA"
        seq_b = "CATACCCA"
        align = aln.make_align(mode="global")
        seq_a_aln, seq_b_aln, _, _, _ = align(seq_a=seq_a, seq_b=seq_b)
        self.assertEqual(seq_a_aln, ["G", "A", "T", "T", "A", "-", "-", "C", "A"])
        self.assertEqual(seq_b_aln, ["C", "A", "-", "T", "A", "C", "C", "C", "A"])


if __name__ == "__main__":
    unittest.main()

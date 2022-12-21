"""Tests for the entropy module."""


import unittest
import entropy


class EntropyTestCase(unittest.TestCase):
    """Test case for the entropy function."""

    def test_no_data(self):
        """Test correct behavior when no data is provided."""
        self.assertEqual(entropy.entropy([]), 0.0)

    def test_zero_entropy(self):
        """Test correct behavior when the data has zero entropy."""
        self.assertEqual(entropy.entropy([0, 0, 0]), 0.0)

    def test_one_bit_entropy(self):
        """Test for correct computation of one bit of entropy."""
        self.assertEqual(entropy.entropy([0, 1]), 1.0)

    def test_string_input(self):
        """Test string input."""
        self.assertAlmostEqual(entropy.entropy("GATTACA"), 1.84, places=2)


if __name__ == "__main__":
    unittest.main()

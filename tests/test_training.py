"""
Basic training utilities test.
"""

import unittest

from utils.seed import SeedManager
from utils.device import DeviceManager


class TestTrainingUtilities(unittest.TestCase):

    def test_seed(self):

        SeedManager.set_seed(42)

        self.assertTrue(True)

    def test_device(self):

        device = DeviceManager.get_device()

        self.assertIn(
            device,
            [
                "cpu",
                "cuda",
            ],
        )


if __name__ == "__main__":

    unittest.main()
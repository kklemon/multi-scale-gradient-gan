import unittest

import torch

from msg_gan.utils import create_multi_scale_image_grid


class TestUtils(unittest.TestCase):
    def test_create_multi_scale_grid(self):
        log2_min_size = 2  # 2^2 = 4 -> min size
        log2_max_size = 6  # 2^6 = 64 -> max size

        batch_size = 4

        random_multi_scale_batches = []
        for log2_size in range(log2_min_size, log2_max_size + 1):
            size = 2 ** log2_size
            random_multi_scale_batches.append(torch.randn(batch_size, 3, size, size))

        create_multi_scale_image_grid(random_multi_scale_batches)

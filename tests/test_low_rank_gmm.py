import unittest
import numpy.testing as npt
import torch
from gmm import LowRankGMM


class TestLowRankGMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_log_prob(self):
        batch_size = 17
        num_components = 3
        rank = 2
        dim = 7
        gmm = LowRankGMM(dim=dim, rank=rank, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        label = torch.randn(batch_size, dim)
        nll = gmm(out_pred, label)
        npt.assert_array_equal(len(nll), batch_size)

    def test_sample(self):
        batch_size = 17
        num_components = 3
        rank = 2
        dim = 7
        gmm = LowRankGMM(dim=dim, rank=rank, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        samples = gmm.sample(out_pred, torch.Size([100,]))
        npt.assert_array_equal(samples.shape, [100, batch_size, dim])

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()

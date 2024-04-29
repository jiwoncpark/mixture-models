import unittest
import numpy.testing as npt
import torch
from mm import FullRankGMM


class TestLowRankGMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_forward(self):
        batch_size = 17
        num_components = 3
        dim = 7
        gmm = FullRankGMM(dim=dim, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        label = torch.randn(batch_size, dim)
        nll = gmm(out_pred, label)
        npt.assert_array_equal(len(nll), batch_size)

    def test_forward_univariate(self):
        batch_size = 17
        num_components = 3
        dim = 1
        gmm = FullRankGMM(dim=dim, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        label = torch.randn(batch_size, dim)
        nll = gmm(out_pred, label)
        npt.assert_array_equal(len(nll), batch_size)

    def test_sample(self):
        batch_size = 17
        num_components = 3
        dim = 7
        gmm = FullRankGMM(dim=dim, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        samples = gmm.sample(out_pred, torch.Size([100,]))
        npt.assert_array_equal(samples.shape, [100, batch_size, dim])

    # def test_cdf(self):
    #     batch_size = 17
    #     num_components = 3
    #     dim = 7
    #     gmm = FullRankGMM(dim=dim, num_components=num_components)
    #     out_pred = torch.randn(batch_size, gmm.out_dim)
    #     formatted = gmm._format(out_pred)
    #     gmm_torch, (mix_torch, comp_torch) = gmm._get_distribution(**formatted)
    #     label = torch.randn(batch_size, dim)
    #     # cdf = gmm_torch.cdf(label)
    #     comp_cdf = comp_torch.cdf(label)
    #     print(comp_cdf)

    def get_marginal_nll(self, gmm, out_pred, label, keep_idx):
        label = label[..., keep_idx]
        nll = gmm.get_marginal_nll(out_pred, label, keep_idx=keep_idx)
        return nll

    def get_conditional_nll(self, gmm, out_pred, label, keep_idx):
        cond_idx = [d for d in range(gmm.dim) if d not in keep_idx]
        condition_vals = label[..., cond_idx]
        nll = gmm.get_conditional_nll(out_pred, label[..., keep_idx], condition_vals, condition_idx=cond_idx)
        return nll

    def test_get_marginal_nll(self):
        batch_size = 17
        num_components = 3
        dim = 7
        label = torch.randn(batch_size, dim)
        gmm = FullRankGMM(dim=dim, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        nll = self.get_marginal_nll(gmm, out_pred, label, keep_idx=[0, 2])
        npt.assert_array_equal(len(nll), batch_size)

    def test_get_conditional_nll(self):
        batch_size = 17
        num_components = 8
        dim = 5
        gmm = FullRankGMM(dim=dim, num_components=num_components)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        keep_idx = [0, 2, 3]
        cond_idx = [1, 4]
        label = torch.randn(batch_size, dim)
        cond_nll = self.get_conditional_nll(gmm, out_pred, label, keep_idx)
        npt.assert_array_equal(len(cond_nll), batch_size)
        margin_nll = self.get_marginal_nll(gmm, out_pred, label, cond_idx)
        nll = gmm(out_pred, label)
        # print(nll - (cond_nll + margin_nll))
        npt.assert_array_almost_equal(nll, cond_nll + margin_nll, decimal=3)

    def test_get_conditional_nll_is_circular(self):
        batch_size = 17
        num_components = 8
        dim = 5
        gmm = FullRankGMM(dim=dim, num_components=num_components, is_circular=True)
        out_pred = torch.randn(batch_size, gmm.out_dim)
        keep_idx = [0, 2, 3]
        cond_idx = [1, 4]
        label = torch.randn(batch_size, num_components, dim)
        cond_nll = self.get_conditional_nll(gmm, out_pred, label, keep_idx)
        npt.assert_array_equal(len(cond_nll), batch_size)
        margin_nll = self.get_marginal_nll(gmm, out_pred, label, cond_idx)
        nll = gmm(out_pred, label)
        # print(nll - (cond_nll + margin_nll))
        npt.assert_array_almost_equal(nll, cond_nll + margin_nll, decimal=3)

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()

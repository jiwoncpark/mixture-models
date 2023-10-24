from torch import Tensor
from typing import Callable, Optional
import torch.distributions as D
import torch
import math


class LowRankGMM(torch.nn.Module):
    def __init__(self, dim: int, rank: int, num_components: int):
        """
        Mixture of multivariate Gaussians with covariance matrices having a low-rank form
        parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag
        
        """
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_components = num_components
        self.shapes = [1, self.dim, self.dim*self.rank, self.dim]
        self.out_dim = sum(self.shapes)*self.num_components

    def _format(self, out_pred: Tensor):        
        out_pred = out_pred.reshape(-1, self.num_components, sum(self.shapes))
        pis, mus, cov_factors, cov_diags = torch.split(out_pred, self.shapes, dim=-1)
        pis = pis.squeeze(-1)  # [batch_size, num_components,]
        cov_factors = cov_factors.reshape(-1, self.num_components, self.dim, self.rank)
        cov_diags = cov_diags.reshape(-1, self.num_components, self.dim).exp()
        out_dict = {
            'pis': pis,
            'mus': mus,
            'cov_factors': cov_factors,
            'cov_diags': cov_diags
        }
        return out_dict

    def _get_distribution(self, pis: Tensor, mus: Tensor, cov_factors: Tensor, cov_diags: Tensor):
        """
        Parameters
        ----------
        pis : Tensor
            Categorical logits of shape `[batch_size, num_components,]`
        mus : Tensor
            MVN loc parameter of shape `[batch_size, num_components, dim]`
        cov_factors : Tensor
            MVN scale parameter of shape `[batch_size, num_components, dim, rank]`
        cov_diags : Tensor
            MVN diagonal parameter of shape `[batch_size, num_components, dim]`
        
        """
        mix = D.Categorical(logits=pis)
        comp = D.LowRankMultivariateNormal(mus, cov_factors, cov_diags)
        gmm_dist = D.MixtureSameFamily(mix, comp)
        return gmm_dist
    
    def get_distribution(self, out_pred):
        formatted = self._format(out_pred)
        return self._get_distribution(**formatted)

    def forward(self, pred: Tensor, label: Tensor):
        """
        Parameters
        ----------
        pred : Tensor
            Predicted distributional params of shape `[batch_size, out_dim]`
        labels : Tensor
            Labels of shape `[batch_size, dim]`
        """
        gmm_dist = self.get_distribution(pred)
        return -gmm_dist.log_prob(label)
    
    def sample(self, pred: Tensor, size: torch.Size):
        gmm_dist = self.get_distribution(pred)
        return gmm_dist.sample(size)

    def string(self):
        return "mixture of low-rank multivariate Gaussians"
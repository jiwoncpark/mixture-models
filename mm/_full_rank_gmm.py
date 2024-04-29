from torch import Tensor
from typing import Callable, Optional
import torch.distributions as D
import torch
from mm._utils import marginalize_1d, marginalize_2d


class FullRankGMM(torch.nn.Module):
    def __init__(
        self, dim: int, num_components: int, is_circular: bool = False,
        to_rad: float = 1.0):
        """
        Mixture of multivariate Gaussians

        Parameters
        ----------
        to_rad : float
            Conversion factor to rad, expected of asin/sin if is_circular
        """
        super().__init__()
        self.dim = dim
        self.num_components = num_components
        self.tril_idx = torch.tril_indices(self.dim, self.dim,
                                           offset=0)  # lower-triangular idx
        self.tril_len = len(self.tril_idx[0])
        self.shapes = [1, self.dim, self.tril_len]
        self.out_dim = sum(self.shapes)*self.num_components
        self.is_circular = is_circular
        self.to_rad = to_rad

    def _format(self, out_pred: Tensor):
        out_pred = out_pred.reshape(-1, self.num_components, sum(self.shapes))
        pis, mus, scale_trils = torch.split(out_pred, self.shapes, dim=-1)
        pis = pis.squeeze(-1)  # [batch_size, num_components,]
        batch_size = out_pred.shape[0]
        tril = torch.zeros([batch_size, self.num_components, self.dim, self.dim]).to(out_pred.device)
        tril[:, :, self.tril_idx[0], self.tril_idx[1]] = scale_trils
        log_diag_tril = torch.diagonal(tril, offset=0, dim1=-2, dim2=-1)  # [batch_size, dim]
        tril[:, :, torch.eye(self.dim, dtype=bool)] = log_diag_tril.exp()
        # precision_matrix = torch.bmm(tril, torch.transpose(tril, -2, -1))  # [batch_size, dim, dim]
        out_dict = {
            'pis': pis,
            'loc': mus,
            'scale_tril': tril,
        }
        return out_dict

    def get_covariance_matrix(self, scale_tril):
        """
        Return loc parameter

        Returns
        -------
        scale_tril : Tensor
            scale_tril param with shape [batch_size, num_components, dim, dim]
        """
        cov_mat = scale_tril @ scale_tril.mT  # L L^T
        return cov_mat

    def _get_distribution(self, pis: Tensor, **kwargs):
        """
        Parameters
        ----------
        pis : Tensor
            Categorical logits of shape `[batch_size, num_components,]`
        mus : Tensor
            MVN loc parameter of shape `[batch_size, num_components, dim]`
        scale_tril : Tensor
            MVN scale parameter of shape `[batch_size, num_components, scale_tril]`

        """
        mix = D.Categorical(logits=pis)
        comp = D.MultivariateNormal(**kwargs)
        gmm_dist = D.MixtureSameFamily(mix, comp)
        return (gmm_dist, (mix, comp))

    def get_distribution(self, out_pred):
        formatted = self._format(out_pred)
        return self._get_distribution(**formatted)[0]

    def forward(self, pred: Tensor, label: Tensor):
        """
        Parameters
        ----------
        pred : Tensor
            Predicted distributional params of shape `[batch_size, out_dim]`
        labels : Tensor
            Labels of shape `[batch_size, dim]`
            or `[batch_size, num_components, dim]` if is_circular
        """
        formatted = self._format(pred)
        gmm_dist, (mix, comp) = self._get_distribution(**formatted)

        if self.is_circular:
            return self.get_circular_nll(mix, comp, label)
        else:
            return -gmm_dist.log_prob(label)

    def get_circular_nll(self, mix, comp, label):
        log_prob_comp = comp.log_prob(label)  # [batch_size, num_components]
        log_prob = (mix.logits + log_prob_comp).logsumexp(-1)
        return -log_prob

    # def get_cdf(self, pred: Tensor, label: Tensor):
    #     """
    #     Parameters
    #     ----------
    #     pred : Tensor
    #         Predicted distributional params of shape `[batch_size, out_dim]`
    #     labels : Tensor
    #         Labels of shape `[batch_size, dim]`
    #     """
    #     formatted = self._format(pred)
    #     _, (mix, comp) = self._get_distribution(**formatted)
    #     comp_cdf = comp.cdf(label)  # [batch_size, num_components]

    def sample(self, pred: Tensor, size: torch.Size):
        gmm_dist = self.get_distribution(pred)
        return gmm_dist.sample(size)

    def _format_marginal_params(self, out_pred: Tensor, keep_idx: list):
        orig_params = self._format(out_pred)
        orig_params['loc'] = marginalize_1d(orig_params['loc'], keep_idx)
        scale_trils = orig_params.pop('scale_tril')
        orig_params['covariance_matrix'] = marginalize_2d(
            scale_trils @ scale_trils.mT, keep_idx)
        return orig_params

    def get_marginal_nll(self, out_pred: Tensor, label: Tensor, keep_idx: list):
        formatted = self._format_marginal_params(out_pred, keep_idx)
        gmm_dist, (mix, comp) = self._get_distribution(**formatted)
        if self.is_circular:
            return self.get_circular_nll(mix, comp, label)
        else:
            return -gmm_dist.log_prob(label)

    def get_conditional_nll(self, out_pred: Tensor, label: Tensor, condition_vals: Tensor, condition_idx: list):
        """
        Evaluate p(keep|condition)

        Parameters
        ----------
        condition_vals : Tensor
            Conditioned values of shape `[batch_size, len(condition_idx)]` or
            `[batch_size, num_components, len(condition_idx)]` if is_circular

        """
        keep_idx = [d for d in range(self.dim) if d not in condition_idx]
        formatted = self._format(out_pred)

        marginal_keep_formatted = self._format_marginal_params(
            out_pred, keep_idx)  # params of p(keep)
        marginal_cond_formatted = self._format_marginal_params(
            out_pred, condition_idx)  # params of p(cond)

        if not self.is_circular:
            condition_vals = condition_vals.unsqueeze(1)  # [batch_size, 1, len(condition_idx)]
        covs_keep = marginal_keep_formatted['covariance_matrix']
        covs = formatted['scale_tril'] @ formatted['scale_tril'].mT
        covs_keep_cond = marginalize_2d(
            covs, keep_idx=keep_idx, keep_idx_last_dim=condition_idx)
        covs_cond_inv = torch.inverse(marginal_cond_formatted['covariance_matrix'])

        # Define new mus, convs
        mus_cond = marginal_keep_formatted['loc'].unsqueeze(-1) + (
            covs_keep_cond @ covs_cond_inv) @ (
            condition_vals - marginal_cond_formatted['loc']).unsqueeze(-1)
        # [batch_size, num_components, len(keep_idx), 1]
        mus_cond = mus_cond.squeeze(-1)  # [batch_size, num_components, len(keep_idx)]
        covs_cond = covs_keep - covs_keep_cond @ covs_cond_inv @ covs_keep_cond.mT
        comp = D.MultivariateNormal(mus_cond, covariance_matrix=covs_cond)

        # Define new pis
        mix = D.Categorical(logits=formatted['pis'])
        marginal_cond_comp = D.MultivariateNormal(
            loc=marginal_cond_formatted['loc'],  # [batch_size, num_components, len(condition_idx)]
            precision_matrix=covs_cond_inv)  # # [batch_size, num_components, len(condition_idx), len(condition_idx)]
        marginal_cond_gmm = D.MixtureSameFamily(mix, marginal_cond_comp)
        if self.is_circular:
            marginal_cond_log_prob = self.get_circular_nll(
                mix, marginal_cond_comp, condition_vals)
        else:
            marginal_cond_log_prob = -marginal_cond_gmm.log_prob(
                condition_vals.squeeze(1))

        new_log_pis = mix.logits + marginal_cond_comp.log_prob(
            condition_vals) - marginal_cond_log_prob.unsqueeze(-1)
        # [batch_size, num_components]

        # Final cond prob
        mix = D.Categorical(logits=new_log_pis)
        gmm_dist = D.MixtureSameFamily(mix, comp)
        if self.is_circular:
            return self.get_circular_nll(mix, comp, label)
        else:
            return -gmm_dist.log_prob(label)

    def string(self):
        return "mixture of low-rank multivariate Gaussians"

import torch


def marginalize_1d(orig_tensor, keep_idx):
    keep_idx = torch.tensor(keep_idx)
    marginalized = torch.index_select(
        orig_tensor, 
        dim=-1, index=keep_idx)
    return marginalized


def marginalize_2d(orig_tensor, keep_idx, keep_idx_last_dim: list = None):
    keep_idx = torch.tensor(keep_idx)
    if keep_idx_last_dim is None:
        keep_idx_last_dim = keep_idx
    else:
        keep_idx_last_dim = torch.tensor(keep_idx_last_dim)
    marginalized = torch.index_select(
        torch.index_select(orig_tensor, dim=-2, index=keep_idx), 
        dim=-1, index=keep_idx_last_dim)
    return marginalized
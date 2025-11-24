# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


# from https://github.com/jarridrb/DEM/blob/main/dem/utils/data_utils.py
def remove_mean(samples, n_particles, spatial_dim):
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, spatial_dim)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, spatial_dim)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


def is_freemean(samples, n_particles, spatial_dim, atol=1e-5):
    mean = samples.view(-1, n_particles, spatial_dim).mean(1)
    return torch.allclose(mean, torch.zeros_like(mean), atol=atol)

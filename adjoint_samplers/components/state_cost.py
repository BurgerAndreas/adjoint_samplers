# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


class GradStateCost:
    def __init__(self):
        pass

    def __call__(self, t, x):
        raise NotImplementedError()


class ZeroGradStateCost(GradStateCost):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, t, x):
        return torch.zeros_like(x)

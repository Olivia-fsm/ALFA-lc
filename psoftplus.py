import warnings

import torch
import math
import functools
from typing import Callable, Optional, Tuple, Union


class ParametricSoftplus(torch.nn.Module):
    # Adapted from:
    # https://discuss.pytorch.org/t/learnable-parameter-in-softplus/60371/2
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#PReLU
    def __init__(self,
                 init_beta: float = 10.0,
                 threshold: float = 20.0,
                 use_curvature_inner=False,
                 use_curvature_outer=False):
        '''For numerical stability the implementation reverts to the linear function when inputxβ>threshold.'''
        super().__init__()
        assert init_beta > 0.0
        assert threshold >= 0.0
        if 0.0 == threshold:
            warnings.warn("This is simply going to be relu")

        # parameterize in terms of log in order to keep beta > 0
        # self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log(), requires_grad=False)
        # self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log(), requires_grad=True)
        if use_curvature_outer and not use_curvature_inner:
            self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log(), requires_grad=True)
        else:
            self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log(), requires_grad=False)
        self.threshold = threshold
        self.register_buffer('offset', torch.log(torch.tensor(2.)), persistent=False)
        self.eps = 1e-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        beta = self.log_beta.exp()
        beta_x = (beta + self.eps) * x
        y = (torch.nn.functional.softplus(beta_x, beta=1.0, threshold=self.threshold) - self.offset) / (beta + self.eps)
        return y
    
    
## Activation Functions for Curvature Analysis ##
def get_activation_obj(activation_name: str) -> torch.nn.Module:
    if activation_name == 'parametric_softplus':
        activation_obj = ParametricSoftplus
    elif activation_name == 'softplus': # mimic ReLU with large beta
        activation_obj = functools.partial(torch.nn.Softplus, beta=1e3)
    elif activation_name == 'low_softplus':
        activation_obj = functools.partial(torch.nn.Softplus, beta=10)
    else:
        raise ValueError(f"unknown activation_name {activation_name}")
    return activation_obj

## Other config layers in lcnn ##
class ApplyOverHalves(torch.nn.Module):
    def __init__(self, f1: Callable, f2: Callable):
        super().__init__()
        self.f1 = f1
        self.f2 = f2

    def __call__(self, x: torch.Tensor):
        assert 4 == x.ndim
        assert 0 == x.shape[1] % 2
        dim = int(x.shape[1] / 2)
        x = torch.cat((self.f1(x[:, :dim, :, :]),
                       self.f2(x[:, dim:, :, :])), axis=1)
        return x


class ChannelRepeatOnce(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        assert 4 == x.ndim
        return x.repeat(1, 2, 1, 1) / math.sqrt(2)


class SumHalves(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        assert 4 == x.ndim
        assert 0 == x.shape[1] % 2
        dim = int(x.shape[1] / 2)
        x = x[:, :dim, :, :] + x[:, dim:, :, :]
        return 0.5 * x


class RowView(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

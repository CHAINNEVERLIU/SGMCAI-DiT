import torch
import torch.nn as nn

from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class DiffusionBase(nn.Module):
    def __init__(self) -> None:
        super(DiffusionBase, self).__init__()

    @abstractmethod
    def q_sample(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Tensor:
        pass

    @torch.no_grad()
    def p_sample(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("请重构p_sample成员函数!")

    @torch.no_grad()
    def p_sample_loop(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("请重构p_sample_loop成员函数!")

    @torch.no_grad()
    def sample(self) -> Tensor:
        raise NotImplementedError("请重构sample成员函数!")

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass




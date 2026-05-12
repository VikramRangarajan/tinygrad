import tinygrad.nn.torch # noqa
import torch

x = torch.randn(3, device="tiny")
y = torch.randn(3, device="tiny")
x + y

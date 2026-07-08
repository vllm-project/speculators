import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual block with optional dimensionality reduction.

    When num_condition > 0, the input dimension is
    hidden_size * (num_condition + 1) and the block projects down to
    hidden_size via both its linear and residual paths.
    """

    def __init__(self, hidden_size: int, num_condition: int = 0):
        super().__init__()
        in_features = hidden_size * (num_condition + 1)
        self.linear = nn.Linear(in_features, hidden_size)
        self.res_connection: nn.Module
        if num_condition > 0:
            self.res_connection = nn.Linear(in_features, hidden_size)
        else:
            self.res_connection = nn.Identity()
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_connection(x) + self.act(self.linear(x))

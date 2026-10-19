import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    _HAS_KERNEL = True
except Exception:  # noqa: BLE001
    mamba_chunk_scan_combined = None
    _HAS_KERNEL = False

_WARNED_NO_KERNEL = False


def _warn_if_gpu_without_kernel(x: torch.Tensor) -> None:
    global _WARNED_NO_KERNEL
    if x.is_cuda and not _HAS_KERNEL and not _WARNED_NO_KERNEL:
        logger.warning(
            "mamba_ssm not installed: MambaDraft is running the slow pure-PyTorch "
            "SSD reference scan on GPU. Install mamba-ssm for production speed."
        )
        _WARNED_NO_KERNEL = True


class _RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


def _ssd_reference(x, dt, A, B, C, D, initial_states):
    b, L, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    rep = h // g
    B = B.repeat_interleave(rep, dim=2)
    C = C.repeat_interleave(rep, dim=2)
    state = initial_states
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t] * A)
        dBx = dt[:, t][..., None, None] * (x[:, t][..., None] * B[:, t][:, :, None, :])
        state = dA[..., None, None] * state + dBx
        y = (state * C[:, t][:, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
        ys.append(y)
    return torch.stack(ys, dim=1)


class MambaDraftMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        conv_kernel: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.d_inner = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.n_groups, self.d_state = n_groups, d_state
        self.conv_kernel = conv_kernel
        self.conv_dim = self.d_inner + 2 * n_groups * d_state

        self.in_proj = nn.Linear(
            hidden_size,
            2 * self.d_inner + 2 * n_groups * d_state + num_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.norm = _RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, initial_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, L, _ = hidden_states.shape
        gate, xBC, dt = torch.split(
            self.in_proj(hidden_states),
            [self.d_inner, self.conv_dim, self.num_heads],
            dim=-1,
        )
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        x = x.reshape(b, L, self.num_heads, self.head_dim)
        B = B.reshape(b, L, self.n_groups, self.d_state)
        C = C.reshape(b, L, self.n_groups, self.d_state)
        A = -torch.exp(self.A_log.float())

        if _HAS_KERNEL and x.is_cuda:
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                chunk_size=L,
                D=self.D,
                dt_bias=self.dt_bias,
                initial_states=initial_states,
                dt_softplus=True,
            )
        else:
            _warn_if_gpu_without_kernel(x)
            if initial_states is None:
                initial_states = torch.zeros(
                    b,
                    self.num_heads,
                    self.head_dim,
                    self.d_state,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            dt = F.softplus(dt.float() + self.dt_bias.float())
            y = _ssd_reference(
                x.float(),
                dt,
                A,
                B.float(),
                C.float(),
                self.D.float(),
                initial_states.float(),
            )
            y = y.to(hidden_states.dtype)

        y = self.norm(y.reshape(b, L, self.d_inner), gate)
        return self.out_proj(y)

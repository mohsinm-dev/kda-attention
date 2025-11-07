from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Tuple


# -----------------------------
# Math (paper, per head):
#   S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
#   o_t = S_t^T q_t
# α_t is a fine-grained diagonal decay (per feature), values in (0,1).
# β_t is a non-negative "learning rate".
# -----------------------------


@dataclass
class KDAState:
    """Fast-weight memory state per batch and head.

    S: (B, H, d_k, d_v)
    """
    S: torch.Tensor


class KDAUpdate(nn.Module):
    """Exact KDA fast-weight update, vectorized over batch & heads.

    Implements one step:
        S_t = (I - β k k^T) Diag(α) S_{t-1} + β k v^T
        o_t = S_t^T q

    Shapes
    ------
    S    : (B, H, d_k, d_v)
    k,q  : (B, H, d_k)
    v    : (B, H, d_v)
    α    : (B, H, d_k)  in (0,1)
    β    : (B, H, 1)    >= 0
    """

    def forward(
        self,
        S: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # (1) S_decay = Diag(α) S_{t-1}
        S_decay = alpha.unsqueeze(-1) * S                      # (B,H,d_k,d_v)

        # (2) (I - β k k^T) S_decay  =  S_decay - β [k (k^T S_decay)]
        kT_S = torch.einsum("bhd,bhdv->bhv", k, S_decay)       # (B,H,d_v)
        kkT_S = k.unsqueeze(-1) * kT_S.unsqueeze(-2)           # (B,H,d_k,d_v)
        left = S_decay - beta.unsqueeze(-1) * kkT_S            # (B,H,d_k,d_v)

        # (3) + β k v^T
        kvT = k.unsqueeze(-1) * v.unsqueeze(-2)                # (B,H,d_k,d_v)
        S_new = left + beta.unsqueeze(-1) * kvT                # (B,H,d_k,d_v)

        # (4) Readout: o = S_new^T q
        o = torch.einsum("bhdv,bhd->bhv", S_new, q)            # (B,H,d_v)
        return S_new, o


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, D = x.shape
    assert D % n_heads == 0, "D must be divisible by n_heads."
    Dh = D // n_heads
    return x.view(B, T, n_heads, Dh)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, T, H, Dh = x.shape
    return x.contiguous().view(B, T, H * Dh)


class KDAAttention(nn.Module):
    """Multi-head KDA layer with projection heads and α/β gates.

    This is the research-minimal module: projection layers, exact update,
    'scan' (training) and 'recurrent' (generation) modes.

    α = sigmoid(W_α x) -> (0,1)^{d_k}
    β = softplus(W_β x) -> [0,∞)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int | None = None,
        out_proj: bool = True,
        alpha_bias: float = 2.0,
        beta_bias: float = 0.0,
    ) -> None:
        super().__init__()

        if d_head is None:
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
            d_head = d_model // n_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_k = d_head
        self.d_v = d_head

        # Projections
        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.w_k = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.w_v = nn.Linear(d_model, n_heads * d_head, bias=True)

        # Gates
        self.w_alpha = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)

        # Output projection
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=True) if out_proj else nn.Identity()

        # Stable init for gates
        nn.init.constant_(self.w_alpha.bias, alpha_bias)
        nn.init.constant_(self.w_beta.bias, beta_bias)

        self._upd = KDAUpdate()

    def init_state(self, batch_size: int, device=None, dtype=None) -> KDAState:
        S = torch.zeros(batch_size, self.n_heads, self.d_k, self.d_v, device=device, dtype=dtype)
        return KDAState(S=S)

    def _project(self, x: torch.Tensor):
        q = _split_heads(self.w_q(x), self.n_heads)  # (B,T,H,dk)
        k = _split_heads(self.w_k(x), self.n_heads)  # (B,T,H,dk)
        v = _split_heads(self.w_v(x), self.n_heads)  # (B,T,H,dv)
        alpha = torch.sigmoid(_split_heads(self.w_alpha(x), self.n_heads))   # (B,T,H,dk)
        beta = F.softplus(self.w_beta(x)).unsqueeze(-1)                      # (B,T,H,1)
        return q, k, v, alpha, beta

    def step(self, x_t: torch.Tensor, state: KDAState) -> tuple[torch.Tensor, KDAState]:
        """One-timestep recurrent update.

        x_t : (B, d_model)
        returns: (B, d_model), new state
        """
        x_t = x_t.unsqueeze(1)  # (B,1,D)
        q, k, v, alpha, beta = self._project(x_t)
        q, k, v, alpha, beta = q[:, 0], k[:, 0], v[:, 0], alpha[:, 0], beta[:, 0]
        S_new, o = self._upd(state.S, k, v, q, alpha, beta)
        y = _merge_heads(o.unsqueeze(1))[:, 0]
        y = self.w_o(y)
        return y, KDAState(S=S_new)

    def forward(
        self,
        x: torch.Tensor,
        state: KDAState | None = None,
        mode: Literal["scan", "recurrent"] = "scan",
    ) -> tuple[torch.Tensor, KDAState]:
        B, T, D = x.shape
        q, k, v, alpha, beta = self._project(x)
        S = state.S if state is not None else self.init_state(B, x.device, x.dtype).S

        if mode == "recurrent":
            assert T == 1, "mode='recurrent' requires T==1; use .step() for single tokens."
            S, o_t = self._upd(S, k[:, 0], v[:, 0], q[:, 0], alpha[:, 0], beta[:, 0])
            y = self.w_o(_merge_heads(o_t.unsqueeze(1))[:, 0]).unsqueeze(1)
            return y, KDAState(S=S)

        outs = []
        for t in range(T):
            S, o_t = self._upd(S, k[:, t], v[:, t], q[:, t], alpha[:, t], beta[:, t])
            outs.append(o_t)
        O = torch.stack(outs, dim=1)                      # (B,T,H,dv)
        y = self.w_o(_merge_heads(O))                    # (B,T,D)
        return y, KDAState(S=S)

from __future__ import annotations
from typing import Tuple
import torch

def kda_chunk_naive(
    S: torch.Tensor,      # (B,H,d_k,d_v)
    K: torch.Tensor,      # (B,H,C,d_k)
    V: torch.Tensor,      # (B,H,C,d_v)
    Q: torch.Tensor,      # (B,H,C,d_k)
    alpha: torch.Tensor,  # (B,H,C,d_k) in (0,1)
    beta: torch.Tensor,   # (B,H,C,1)   >= 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unroll the exact KDA recurrence across a chunk of length C.

    Per step (paper):
        S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
        o_t = S_t^T q_t
    """
    if S.dim() != 4:
        raise ValueError("S must be (B,H,d_k,d_v).")
    if K.dim() != 4 or V.dim() != 4 or Q.dim() != 4:
        raise ValueError("K,V,Q must be (B,H,C,dk/dv).")
    if alpha.dim() != 4 or beta.dim() != 4:
        raise ValueError("alpha (B,H,C,dk), beta (B,H,C,1).")

    B, H, C, dk = K.shape
    _, _, _, dv = V.shape

    O = []
    S_curr = S
    for t in range(C):
        k_t = K[:, :, t, :]           # (B,H,dk)
        v_t = V[:, :, t, :]           # (B,H,dv)
        q_t = Q[:, :, t, :]           # (B,H,dk)
        a_t = alpha[:, :, t, :]       # (B,H,dk)
        b_t = beta[:, :, t, :]        # (B,H,1)

        S_decay = a_t.unsqueeze(-1) * S_curr
        kT_S   = torch.einsum("bhd,bhdv->bhv", k_t, S_decay)
        kkT_S  = k_t.unsqueeze(-1) * kT_S.unsqueeze(-2)
        left   = S_decay - b_t.unsqueeze(-1) * kkT_S

        kvT    = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        S_curr = left + b_t.unsqueeze(-1) * kvT

        o_t    = torch.einsum("bhdv,bhd->bhv", S_curr, q_t)
        O.append(o_t)

    O = torch.stack(O, dim=2)         # (B,H,C,dv)
    return O, S_curr

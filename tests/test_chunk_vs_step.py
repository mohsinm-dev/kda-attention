from __future__ import annotations
import torch
from kda_attn.kda import KDAUpdate
from kda_attn.chunk_naive import kda_chunk_naive

def test_chunk_naive_equals_unrolled_steps() -> None:
    torch.manual_seed(0)
    B, H, C, dk, dv = 2, 3, 5, 7, 11
    S0 = torch.randn(B,H,dk,dv)
    K  = torch.randn(B,H,C,dk)
    V  = torch.randn(B,H,C,dv)
    Q  = torch.randn(B,H,C,dk)
    alpha = torch.sigmoid(torch.randn(B,H,C,dk))
    beta  = torch.rand(B,H,C,1)

    O_chunk, S_chunk = kda_chunk_naive(S0.clone(), K, V, Q, alpha, beta)

    upd = KDAUpdate()
    S = S0.clone()
    O_steps = []
    for t in range(C):
        S, o_t = upd(S, K[:,:,t,:], V[:,:,t,:], Q[:,:,t,:], alpha[:,:,t,:], beta[:,:,t,:])
        O_steps.append(o_t)
    O_steps = torch.stack(O_steps, dim=2)

    assert torch.allclose(O_chunk, O_steps, atol=1e-6)
    assert torch.allclose(S_chunk, S, atol=1e-6)

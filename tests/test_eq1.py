from __future__ import annotations
import torch
from kda_attn.kda import KDAUpdate

def test_kda_matches_paper_equation() -> None:
    torch.manual_seed(0)
    B, H, dk, dv = 2, 3, 5, 7
    S  = torch.randn(B,H,dk,dv)
    k  = torch.randn(B,H,dk)
    v  = torch.randn(B,H,dv)
    q  = torch.randn(B,H,dk)
    alpha = torch.sigmoid(torch.randn(B,H,dk))
    beta  = torch.rand(B,H,1)  # stand-in (module uses softplus output)

    # Reference:
    # S_t = (I - β kk^T) Diag(α) S + β k v^T ; o = S_t^T q
    S_decay = alpha.unsqueeze(-1) * S
    kT_S    = torch.einsum("bhd,bhdv->bhv", k, S_decay)
    kkT_S   = k.unsqueeze(-1) * kT_S.unsqueeze(-2)
    left    = S_decay - beta.unsqueeze(-1) * kkT_S
    ref_S   = left + beta.unsqueeze(-1) * (k.unsqueeze(-1) * v.unsqueeze(-2))
    ref_o   = torch.einsum("bhdv,bhd->bhv", ref_S, q)

    upd = KDAUpdate()
    S_new, o = upd(S, k, v, q, alpha, beta)

    assert torch.allclose(S_new, ref_S, atol=1e-6)
    assert torch.allclose(o,     ref_o, atol=1e-6)

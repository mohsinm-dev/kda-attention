from __future__ import annotations
import torch
from kda_attn.kda import KDAAttention

def main() -> None:
    torch.manual_seed(0)
    B, T, D, H = 2, 16, 64, 4
    x = torch.randn(B, T, D)
    layer = KDAAttention(d_model=D, n_heads=H)
    y, state = layer(x, state=None, mode="scan")
    print(y.shape, state.S.shape)

if __name__ == "__main__":
    main()

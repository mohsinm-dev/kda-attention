from __future__ import annotations
import torch
from kda_attn.kda import KDAAttention

def main() -> None:
    torch.manual_seed(0)
    B, D, H = 2, 64, 4
    layer = KDAAttention(d_model=D, n_heads=H)
    state = layer.init_state(B)

    x_t = torch.randn(B, D)
    for _ in range(3):
        y_t, state = layer.step(x_t, state)
        print("y_t:", y_t.shape)
        x_t = y_t.detach()  # dummy next input

if __name__ == "__main__":
    main()

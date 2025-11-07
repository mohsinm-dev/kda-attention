# kda-attention (research-style)

Minimal PyTorch implementation of **Kimi Delta Attention (KDA)** with an exact, paper-faithful
fast-weight recurrence and a naive chunk unroll for verification.

**Per-head update (paper):**
\[
S_t=(I-\beta_t k_t k_t^{\top})\,\mathrm{Diag}(\alpha_t)\,S_{t-1}+\beta_t k_t v_t^{\top},
\qquad
o_t=S_t^{\top} q_t.
\]

What this repo provides:
- `KDAUpdate`: exact one-step update and readout.
- `KDAAttention`: multi-head layer with projections + α/β gates; scan & recurrent modes.
- `kda_chunk_naive`: chunk unroll that is *mathematically identical* to repeating the step update.
- Tests that prove equality to the equation above and to the unrolled chunk.

> This repo intentionally does **not** implement the paper's custom packed WY/UT kernels;
> those are performance engineering and do not change the model. The API here keeps a clear
> path to add such kernels later.

## Quick start

```python
import torch
from kda_attn import KDAAttention

x = torch.randn(2, 128, 768)
layer = KDAAttention(d_model=768, n_heads=12)
y, state = layer(x, state=None, mode="scan")
print(y.shape)            # (2,128,768)
print(state.S.shape)      # (2,12,64,64) if d_head=64
```

## Tests

```bash
pytest -q
```

## License

Apache-2.0

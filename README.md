# KDA Attention

PyTorch implementation of **Kimi Delta Attention (KDA)** - an efficient attention mechanism with fast-weight recurrence for improved long-context modeling.

## Overview

KDA uses a gated linear recurrence to maintain a fast-weight memory state that evolves over time. The core update equations are:

```
S_t = (I - β_t * k_t * k_t^T) * Diag(α_t) * S_{t-1} + β_t * k_t * v_t^T
o_t = S_t^T * q_t
```

Where:
- `α_t` is a fine-grained diagonal decay factor (values in (0,1))
- `β_t` is a non-negative learning rate
- `S_t` is the fast-weight memory state

## Features

- **KDAUpdate**: Exact one-step recurrence update and readout
- **KDAAttention**: Multi-head attention layer with trainable α/β gates  
- **Dual modes**: Training mode (`scan`) and generation mode (`recurrent`)
- **Chunk processing**: Naive chunk implementation for verification
- **Comprehensive tests**: Mathematical correctness verification

## Quick Start

### Basic Usage

```python
import torch
from kda_attn import KDAAttention

# Initialize layer
layer = KDAAttention(d_model=768, n_heads=12)

# Training mode - process full sequences
x = torch.randn(2, 128, 768)  # (batch, seq_len, d_model)
y, state = layer(x, state=None, mode="scan")
print(y.shape)            # (2, 128, 768)
print(state.S.shape)      # (2, 12, 64, 64)
```

### Generation Mode

```python
# Initialize state for generation
batch_size = 2
state = layer.init_state(batch_size)

# Step-by-step processing
for _ in range(10):
    x_t = torch.randn(batch_size, 768)
    y_t, state = layer.step(x_t, state)
    print(y_t.shape)  # (2, 768)
```

## Installation

```bash
git clone https://github.com/mohsinm-dev/kda-attention.git
cd kda-attention
pip install -e .
```

## Testing

Run the test suite to verify correctness:

```bash
pytest tests/ -v
```

Run examples:

```bash
python examples/scan_demo.py      # Training mode demo
python examples/recurrent_demo.py # Generation mode demo
```

## License

Apache-2.0


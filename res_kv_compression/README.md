# Low-Rank + Quantized Residual KV Cache Compression

Research-grade PyTorch framework for transformer KV cache compression:

```text
K ~= K_LR + DQ(K_Q)
V ~= V_LR + DQ(V_Q)
```

The project is intentionally modular so the first implementation can prioritize
correctness, reproducibility, and measurable research behavior before CUDA or
Triton optimization.

## Quick Start

```bash
python run_experiment.py config=configs/tinyllama_int4_rank16.yaml
pytest
```

The default experiment runner only loads HuggingFace models when an evaluation
task requiring model execution is enabled.

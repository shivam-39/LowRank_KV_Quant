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

To visualize results from a completed run, generate plots from the JSON logs:

```bash
python scripts/plot_results.py --run logs/tinyllama_int4_rank16
```

Generation memory experiment:

```bash
python run_experiment.py config=configs/tinyllama_generation_memory.yaml
python scripts/plot_results.py --run logs/tinyllama_generation_memory
```

This generates one token at a time until `model.max_seq_len`, reconstructing
the compressed KV cache before each decode step. The run writes
`generation_memory.jsonl` plus `plots/generation_memory.png`, comparing KV
memory with and without compression as sequence length grows.

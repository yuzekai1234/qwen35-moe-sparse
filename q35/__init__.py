"""Scaled Qwen3.5-MoE sparse pretraining baselines.

Modules:
  models/   editable Qwen3.5-MoE text model (vendored from transformers 5.19.0)
  presets   configs/<name>.json presets -> model config, compute-budget token counts
  flops     per-token parameter / FLOPs accounting
  data      corpus preparation and uint32 token-file loading
  train     pretraining entry point (python -m q35.train / torchrun -m q35.train)
  evaluate  official-style bpb / EvalTime evaluation (torchrun -m q35.evaluate)
"""

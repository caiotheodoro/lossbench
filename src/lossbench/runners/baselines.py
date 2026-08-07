"""Canonical baseline model registry.

Pricing is per 1k tokens in USD at plausible 2026 hosted open-model rates.
These figures are a swappable configuration for benchmarking, not gospel:
update them from the model provider's current pricing page before any
published cost comparison.
"""

from __future__ import annotations

BASELINE_MODELS: dict[str, dict] = {
    "reconforge-1.7b": {
        "model_id": "caiotheodoro/reconforge-recon-lora",
        "cost_per_1k_in": 0.0,
        "cost_per_1k_out": 0.0,
    },
    "qwen3.8-27b": {
        "model_id": "qwen/qwen3.8-27b",
        "cost_per_1k_in": 0.35,
        "cost_per_1k_out": 0.7,
    },
    "nemotron-3.5-lightning": {
        "model_id": "nvidia/nemotron-3.5-lightning",
        "cost_per_1k_in": 0.2,
        "cost_per_1k_out": 0.4,
    },
    "deepseek-v4-flash": {
        "model_id": "deepseek/deepseek-v4-flash",
        "cost_per_1k_in": 0.1,
        "cost_per_1k_out": 0.3,
    },
    "qwen3.8-2.4t-a95b": {
        "model_id": "qwen/qwen3.8-2.4t-a95b",
        "cost_per_1k_in": 0.5,
        "cost_per_1k_out": 1.0,
    },
}

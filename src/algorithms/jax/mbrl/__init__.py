"""Model-based RL training stack for in-agent nightly retraining.

Single-instance port of model-uncertainty-exploration (feat/classic-plan-every,
jax 0.10/flax 0.12) with the multi-seed/multi-config vmap grid batching
stripped, brax support dropped, and PPO training refactored into bounded
chunks so it can run incrementally inside the RlGlue plan() hook. Keep the
module contents in sync with the source repo when retraining semantics change.
"""

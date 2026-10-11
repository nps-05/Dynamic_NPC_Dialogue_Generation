"""Model id used when HF_MODEL is not set.

Revert to the previous default by setting HF_MODEL in .env, or by changing
DEFAULT_HF_MODEL back to PREVIOUS_DEFAULT_MODEL.
"""

# Qwen/Qwen2.5-3B-Instruct — previous default. Higher quality, about 6 GB, slower on CPU.
PREVIOUS_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Smaller instruct model so a CPU demo is usable. About 3 GB.
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

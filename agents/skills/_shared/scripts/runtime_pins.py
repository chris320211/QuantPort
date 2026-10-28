"""Pinned baseline used by method virtualenvs."""

RUNTIME_PACKAGES: tuple[str, ...] = (
    "transformers==4.46.3",
    "accelerate==1.1.1",
    "safetensors==0.4.5",
    "sentencepiece==0.2.0",
    "datasets==3.1.0",
)

# vLLM serve check venv (.venvs/vllm-serve). vLLM brings its own torch and
# transformers, so it never shares a method venv. Bump after a GPU run passes.
SERVE_PACKAGES: tuple[str, ...] = (
    "vllm==0.30.0",
    "datasets>=3.1,<5",
)

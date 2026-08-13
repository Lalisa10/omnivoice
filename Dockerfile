# syntax=docker/dockerfile:1
#
# OmniVoice serving image (Ray Serve / FastAPI HTTP layer).
#
# Build:
#   docker build -t omnivoice-serve .
#
# Run (needs the NVIDIA Container Toolkit on the host):
#   docker run --gpus all -p 8000:8000 \
#     --shm-size=2g \
#     -v omnivoice-hf-cache:/home/omnivoice/.cache/huggingface \
#     -v omnivoice-voices:/home/omnivoice/.cache/omnivoice/voices \
#     -e OMNIVOICE_MODEL=k2-fsa/OmniVoice \
#     omnivoice-serve
#
# The model is downloaded from Hugging Face on first start (mount a volume
# over the HF cache dir so it isn't re-downloaded on every container start).
# If the HF Hub is blocked from your network, also pass
# -e HF_ENDPOINT=https://hf-mirror.com

ARG PYTHON_VERSION=3.12

# ---- builder: resolve deps with uv (incl. the pinned torch/torchaudio cu128
# wheels from pyproject.toml) and install the project into a venv ----------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first (cached separately from source changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Now install the project itself (needs pyproject.toml/uv.lock in the layer
# itself, not just bind-mounted, since this RUN is a separate build step).
COPY pyproject.toml uv.lock README.md ./
COPY omnivoice ./omnivoice
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime: slim image with just the venv + source -----------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# ffmpeg: pydub/audio decoding for non-wav formats.
# libsndfile1: required by soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system omnivoice && useradd --system --gid omnivoice --create-home omnivoice

WORKDIR /app
COPY --from=builder --chown=omnivoice:omnivoice /app/.venv /app/.venv
COPY --chown=omnivoice:omnivoice omnivoice ./omnivoice
COPY --chown=omnivoice:omnivoice serve_config.yaml ./serve_config.yaml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/omnivoice \
    OMNIVOICE_LOG_LEVEL=INFO \
    OMNIVOICE_MODEL=k2-fsa/OmniVoice

# Pre-create with the right owner: an anonymous VOLUME mounted over a path
# that doesn't exist yet in the image is created by the docker daemon as
# root, which the non-root `omnivoice` user then can't write into.
RUN mkdir -p /home/omnivoice/.cache/huggingface /home/omnivoice/.cache/omnivoice/voices \
    && chown -R omnivoice:omnivoice /home/omnivoice/.cache

USER omnivoice

# HF model cache + cloned-voice registry: mount volumes over these to persist
# across container restarts (see `docker run` example above).
VOLUME ["/home/omnivoice/.cache/huggingface", "/home/omnivoice/.cache/omnivoice"]

EXPOSE 8000

CMD ["serve", "run", "/app/serve_config.yaml", "--blocking"]

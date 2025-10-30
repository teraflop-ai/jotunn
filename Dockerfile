FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

ENV UV_COMPILE_BYTECODE=1 

ENV UV_LINK_MODE=copy

ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    build-essential \
    gcc \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    zlib1g-dev

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

ENV HF_HOME=/root/.cache/huggingface

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra all

ENV PATH="/workspace/.venv/bin:$PATH"    

WORKDIR /workspace

CMD ["python3"]
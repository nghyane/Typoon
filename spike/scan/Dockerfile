FROM python:3.12-slim AS bunle-build

# Build bunle Rust extension for linux. Keeps Rust toolchain out of the
# final image; only the wheel comes across.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

COPY vendor/bunle /src/bunle
WORKDIR /src/bunle/python
RUN maturin build --release -o /wheels


# ── Final image ───────────────────────────────────────────────────────────────

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libgomp1 \
    ca-certificates fuse curl \
    && rm -rf /var/lib/apt/lists/*

# tigrisfs (R2 FUSE mount)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then ARCH="amd64"; fi && \
    if [ "$ARCH" = "aarch64" ]; then ARCH="arm64"; fi && \
    VERSION=$(curl -s https://api.github.com/repos/tigrisdata/tigrisfs/releases/latest \
              | grep -o '"tag_name": "[^"]*' | cut -d'"' -f4) && \
    curl -L "https://github.com/tigrisdata/tigrisfs/releases/download/${VERSION}/tigrisfs_${VERSION#v}_linux_${ARCH}.tar.gz" \
         -o /tmp/tigrisfs.tar.gz && \
    tar -xzf /tmp/tigrisfs.tar.gz -C /usr/local/bin/ && \
    rm /tmp/tigrisfs.tar.gz && \
    chmod +x /usr/local/bin/tigrisfs

# Install bunle wheel from build stage
COPY --from=bunle-build /wheels/bunle-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/bunle-*.whl && rm /tmp/bunle-*.whl

# Install typoon workspace packages — scan slice only (no asyncpg, no LLM provider deps).
# typoon-stages pulls typoon-llm transitively but only LLM provider modules are
# tree-shaken at import time; scan never imports typoon.llm.* so the heavy SDKs
# (anthropic, openai) sit unused in site-packages.
COPY wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Container-specific runtime deps not pulled in by typoon packages
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Comic-DETR model (11MB)
COPY container/comic-detr-v4s-int8.onnx /app/models/comic-detr-v4s-int8.onnx

COPY container/main.py       .
COPY container/storyboard.py .
COPY container/startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 8080
CMD ["/startup.sh"]



# ---------------------------------------------------------------------------
# Build stage: bunle Python wheel from source (parallel sibling project).
# Keeping the Rust toolchain out of the final image — only the wheel ships.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS bunle-build

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

COPY vendor/bunle /src/bunle
WORKDIR /src/bunle/python
RUN maturin build --release -o /wheels

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo ca-certificates fuse curl \
    && rm -rf /var/lib/apt/lists/*

# Install tigrisfs
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

# Pre-built wheel — no Rust/maturin needed at image build time.
# To update: maturin build --release --features python (linux/amd64) → copy .whl here.
COPY container/typoon_render-*.whl .
RUN pip install --no-cache-dir typoon_render-*.whl

# bunle wheel — built in the bunle-build stage above.
COPY --from=bunle-build /wheels/bunle-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/bunle-*.whl && rm -rf /tmp/bunle-*.whl

COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY container/main.py .
COPY container/startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 8080
CMD ["/startup.sh"]

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

COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY container/main.py .
COPY container/startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 8080
CMD ["/startup.sh"]

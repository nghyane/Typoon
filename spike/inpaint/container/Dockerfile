# syntax=docker/dockerfile:1
# SPDX-License-Identifier: GPL-3.0-or-later
# ---------------------------------------------------------------------------
# Build: statically-linked Rust binary (musl target)
# ---------------------------------------------------------------------------
FROM rust:1.88-alpine AS builder

RUN apk add --no-cache musl-dev pkgconfig

WORKDIR /build

# Cache deps layer separately
COPY Cargo.toml Cargo.lock ./
RUN mkdir src bin && echo 'fn main(){}' > bin/serve.rs && echo '' > src/lib.rs \
 && cargo build --release --no-default-features --features serve 2>&1 | tail -1 \
 && rm -rf src bin

COPY src/ ./src/
COPY bin/ ./bin/
RUN touch src/lib.rs bin/serve.rs \
 && cargo build --release --bin serve --no-default-features --features serve

# ---------------------------------------------------------------------------
# Runtime: distroless static — no shell, no package manager
# ---------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12

COPY --from=builder /build/target/release/serve /serve
COPY model.safetensors /app/model.safetensors

ENV MODEL_PATH=/app/model.safetensors
ENV PORT=8000
ENV FP16=0

EXPOSE 8000
ENTRYPOINT ["/serve"]

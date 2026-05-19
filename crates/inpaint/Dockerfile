# syntax=docker/dockerfile:1
# SPDX-License-Identifier: GPL-3.0-or-later
# ---------------------------------------------------------------------------- #
# Build stage: statically-linked Linux binary via musl
# ---------------------------------------------------------------------------- #
FROM rust:1.83-alpine AS builder
RUN apk add --no-cache musl-dev pkgconfig

WORKDIR /build
COPY Cargo.toml ./
COPY src/ ./src/
COPY bin/ ./bin/

# Build release binary for linux/amd64 (Cloudflare Container default)
RUN cargo build --release --bin serve

# ---------------------------------------------------------------------------- #
# Runtime stage: distroless — just binary + weights
# ---------------------------------------------------------------------------- #
FROM gcr.io/distroless/static-debian12

COPY --from=builder /build/target/release/serve /serve

# Weights are baked in at container build time from CI artifact.
# Override via MODEL_PATH env for local testing.
COPY model.safetensors /app/model.safetensors

ENV MODEL_PATH=/app/model.safetensors
ENV PORT=8000
ENV FP16=0

EXPOSE 8000

ENTRYPOINT ["/serve"]

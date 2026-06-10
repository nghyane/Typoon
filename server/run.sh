#!/bin/bash
set -a
[ -f .env ] && source .env
set +a
exec go run ./cmd/api "$@"

#!/bin/sh
set -e

mkdir -p /mnt/r2

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
echo "Mounting R2 bucket ${R2_BUCKET_NAME} via FUSE..."
/usr/local/bin/tigrisfs --endpoint "${R2_ENDPOINT}" -f "${R2_BUCKET_NAME}" /mnt/r2 &

until mountpoint -q /mnt/r2; do sleep 0.1; done
echo "R2 mounted at /mnt/r2"

exec uvicorn main:app --host 0.0.0.0 --port 8080 --log-level info

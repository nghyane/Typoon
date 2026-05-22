#!/usr/bin/env bash
# probe.sh — fast container dev loop for typoon-scan.
#
# Steps:
#   1. Build container image with new timestamp tag
#   2. Push to Cloudflare registry (wrangler containers push)
#   3. Update wrangler.toml image tag
#   4. Deploy worker
#   5. Wait for instance restart
#   6. Run benchmark via /debug-scan
#
# Usage: bash workers/scan/probe.sh [pages]
#   pages: number of pages to scan (default 28)

set -euo pipefail
cd "$(dirname "$0")"

ACCOUNT="818e551312970df676abe1a0e61819c7"
IMAGE_NAME="typoon-scan-scancontainer"
TAG="probe$(date +%s)"
PAGES=${1:-28}
SCAN_URL="https://typoon-scan.hoangvananhnghia99.workers.dev"

echo "==> Step 1/6: Build $IMAGE_NAME:$TAG (linux/amd64)"
docker build --platform linux/amd64 --quiet -t "$IMAGE_NAME:$TAG" . > /dev/null

echo "==> Step 2/6: Push to Cloudflare registry"
docker tag "$IMAGE_NAME:$TAG" "registry.cloudflare.com/$ACCOUNT/$IMAGE_NAME:$TAG"
docker push --quiet "registry.cloudflare.com/$ACCOUNT/$IMAGE_NAME:$TAG" > /dev/null

echo "==> Step 3/6: Update wrangler.toml image tag → $TAG"
# Replace any `image = "registry.cloudflare.com/.../typoon-scan-scancontainer:..."`
# or `image = "./Dockerfile"` with the new tagged URI.
python3 -c "
import re, sys
p = 'wrangler.toml'
s = open(p).read()
new_img = 'registry.cloudflare.com/$ACCOUNT/$IMAGE_NAME:$TAG'
s = re.sub(r'image\s*=\s*\"[^\"]*\"', f'image         = \"{new_img}\"', s, count=1)
open(p, 'w').write(s)
"

echo "==> Step 4/6: Deploy worker"
npx wrangler deploy 2>&1 | grep -E "Current Version|Deployed" | head -3

echo "==> Step 5/6: Wait 30s for container rollout"
sleep 30
curl -s --max-time 60 "$SCAN_URL/warm" > /dev/null
echo "warm OK"

if [ -n "${SKIP_BENCH:-}" ]; then
  echo "==> Skip bench (SKIP_BENCH set)"
  exit 0
fi

echo "==> Step 6/6: Benchmark $PAGES pages"
python3 -c "
import json
pages=[{'page_index': i, 'prepared_key': f'prepared/6fd9bac8-3003-4e0b-982e-40ded61b4194/{i:04d}.jpg', 'is_color': False} for i in range($PAGES)]
print(json.dumps({'pages': pages, 'lang_hint': 'ja'}))
" > /tmp/probe-payload.json

T0=$(python3 -c "import time; print(time.time())")
curl -s --max-time 180 -X POST "$SCAN_URL/debug-scan?job_id=999999" \
  -H "Content-Type: application/json" --data-binary @/tmp/probe-payload.json \
  -o /tmp/probe-result.json -w "HTTP:%{http_code}\n"
T1=$(python3 -c "import time; print(time.time())")
ELAPSED=$(python3 -c "print(round($T1 - $T0, 1))")

echo "wall:    ${ELAPSED}s"
python3 -c "
import json
d = json.load(open('/tmp/probe-result.json'))
print('timings:', d.get('timings_ms'))
print('groups :', sum(1 for _ in d.get('scan_keys', [])))
err = d.get('error')
if err: print('ERROR:', err, d.get('detail',''))
"

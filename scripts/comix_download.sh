#!/bin/bash
# Download a comix.to series. Curl-only, no browser needed during download.
#
# Usage:
#   bash scripts/comix_download.sh <series_slug> [output_dir]
#
# Example:
#   bash scripts/comix_download.sh z0yj-ctrlaltresign tests/fixtures/ctrlaltresign
#
# Prerequisites: run `python3 scripts/comix_cookies.py` once to get CF cookies.
set -euo pipefail

SERIES="${1:?Usage: $0 <series_slug> [output_dir]}"
OUTPUT_DIR="${2:-tests/fixtures/$SERIES}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COOKIE_FILE="$SCRIPT_DIR/../cache/comix_cookies.txt"
PARALLEL=10

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"

# ── Ensure cookies ──
if ! python3 "$SCRIPT_DIR/comix_cookies.py" --check > /dev/null 2>&1; then
  echo "Cookies expired or missing. Grabbing new ones..."
  python3 "$SCRIPT_DIR/comix_cookies.py" || exit 1
fi
COOKIES=$(cat "$COOKIE_FILE")

cfetch() { curl -sL "$1" -H "Cookie: $COOKIES" -H "User-Agent: $UA"; }
export -f cfetch
export COOKIES UA

# ── Step 1: Scrape chapter list ──
echo "=== Scraping chapter list for $SERIES ==="
ALL_LINKS=""
page=1
while true; do
  echo -n "  page $page..."
  HTML=$(cfetch "https://comix.to/title/$SERIES?page=$page")

  # Extract chapter hrefs from RSC/HTML
  LINKS=$(echo "$HTML" | grep -oE "/title/$SERIES/[0-9]+-chapter-[0-9]+" | sort -u)
  if [ -z "$LINKS" ]; then
    # Try RSC flight data (escaped in JSON)
    LINKS=$(echo "$HTML" | grep -oE "/title/${SERIES//\//\\/}/[0-9]+-chapter-[0-9]+" | sort -u)
  fi

  COUNT=$(echo "$LINKS" | grep -c chapter || true)
  echo " $COUNT links"

  if [ "$COUNT" -eq 0 ]; then break; fi
  ALL_LINKS="$ALL_LINKS
$LINKS"
  page=$((page + 1))
  # Safety: comix.to query param pagination may not work, break if same data
  if [ "$page" -gt 1 ]; then break; fi
done

# Deduplicate and pick one link per chapter (highest ID = latest upload)
CHAPTERS=$(echo "$ALL_LINKS" | grep -v '^$' | sort -u | python3 -c "
import sys, re
from collections import defaultdict
chapters = defaultdict(list)
for line in sys.stdin:
    line = line.strip()
    m = re.search(r'/(\d+)-chapter-(\d+)', line)
    if m:
        upload_id, ch = int(m.group(1)), int(m.group(2))
        chapters[ch].append((upload_id, line))
for ch in sorted(chapters):
    best = max(chapters[ch], key=lambda x: x[0])
    print(f'{ch}|{best[1]}')
")

# If pagination didn't work, fallback: scrape all pages via the title page HTML
if [ "$(echo "$CHAPTERS" | wc -l)" -lt 5 ]; then
  echo "  Pagination didn't work, scraping from title page..."
  HTML=$(cfetch "https://comix.to/title/$SERIES")
  CHAPTERS=$(echo "$HTML" | grep -oE "/title/$SERIES/[0-9]+-chapter-[0-9]+" | sort -u | python3 -c "
import sys, re
from collections import defaultdict
chapters = defaultdict(list)
for line in sys.stdin:
    line = line.strip()
    m = re.search(r'/(\d+)-chapter-(\d+)', line)
    if m:
        upload_id, ch = int(m.group(1)), int(m.group(2))
        chapters[ch].append((upload_id, line))
for ch in sorted(chapters):
    best = max(chapters[ch], key=lambda x: x[0])
    print(f'{ch}|{best[1]}')
")
fi

TOTAL_CH=$(echo "$CHAPTERS" | grep -c '|' || true)
echo "Found $TOTAL_CH chapters"
echo "$CHAPTERS" > /tmp/comix_chapters.txt

# ── Step 2: Get base URLs (parallel) ──
echo ""
echo "=== Extracting image base URLs ==="
> /tmp/comix_base_urls.txt

get_base() {
  local chap_num="$1" chap_path="$2"
  local base
  base=$(cfetch "https://comix.to${chap_path}" \
    | grep -oE 'https://[^"]+wowpic[^"]+\.webp' | head -1 | sed 's|/[0-9]*\.webp$||')
  if [ -n "$base" ]; then
    echo "$chap_num|$base"
  else
    echo "  Ch.$chap_num FAILED" >&2
  fi
}
export -f get_base

batch=0
while IFS='|' read -r chap_num chap_path; do
  get_base "$chap_num" "$chap_path" >> /tmp/comix_base_urls.txt &
  batch=$((batch + 1))
  [ $((batch % PARALLEL)) -eq 0 ] && wait
done <<< "$CHAPTERS"
wait

sort -t'|' -k1 -n /tmp/comix_base_urls.txt -o /tmp/comix_base_urls.txt
echo "Got $(wc -l < /tmp/comix_base_urls.txt | tr -d ' ') base URLs"

# ── Step 3: Download images (parallel) ──
echo ""
echo "=== Downloading images ==="
mkdir -p "$OUTPUT_DIR"

download_ch() {
  local chap_num="$1" base_url="$2" outdir="$3"
  local dir="$outdir/ch$(printf '%03d' "$chap_num")"
  mkdir -p "$dir"

  # Binary search for page count
  local lo=1 hi=80
  while [ "$lo" -lt "$hi" ]; do
    local mid=$(( (lo + hi + 1) / 2 ))
    local code
    code=$(curl -sI "${base_url}/$(printf '%02d.webp' "$mid")" -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 5)
    [ "$code" = "200" ] && lo=$mid || hi=$((mid - 1))
  done

  for i in $(seq 1 "$lo"); do
    local fn=$(printf '%02d.webp' "$i")
    [ -f "$dir/$fn" ] && continue
    curl -sL "${base_url}/${fn}" -o "$dir/$fn" --max-time 30 &
  done
  wait
  echo "  Ch.$(printf '%02d' "$chap_num"): $lo pages"
}
export -f download_ch

batch=0
while IFS='|' read -r chap_num base_url; do
  download_ch "$chap_num" "$base_url" "$OUTPUT_DIR" &
  batch=$((batch + 1))
  [ $((batch % PARALLEL)) -eq 0 ] && wait
done < /tmp/comix_base_urls.txt
wait

echo ""
echo "=== Done ==="
find "$OUTPUT_DIR" -name '*.webp' | wc -l | xargs -I{} echo "Total: {} pages"
du -sh "$OUTPUT_DIR"

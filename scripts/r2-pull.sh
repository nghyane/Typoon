#!/usr/bin/env bash
# Pull R2 prefix into the local mock filesystem used by container apps.
#
#   bash scripts/r2-pull.sh raw/1/source.zip
#   bash scripts/r2-pull.sh prepared/1/    # pulls everything under prefix
#
# Reads R2 creds from workers/api/.dev.vars so the same secrets used in
# production drive the mock. Files land at $R2_MOUNT (default /tmp/r2-mock)
# preserving the bucket-relative path.

set -euo pipefail
PREFIX="${1:?usage: r2-pull.sh <key-or-prefix>}"
ROOT="${R2_MOUNT:-/tmp/r2-mock}"
DEV_VARS="$(cd "$(dirname "$0")/.." && pwd)/workers/api/.dev.vars"
[ -r "$DEV_VARS" ] || { echo "missing $DEV_VARS"; exit 1; }
AK=$(grep R2_ACCESS_KEY_ID "$DEV_VARS" | cut -d= -f2)
SK=$(grep R2_SECRET_ACCESS_KEY "$DEV_VARS" | cut -d= -f2)
ACC=$(grep R2_ACCOUNT_ID "$DEV_VARS" | cut -d= -f2)
BUCKET=$(grep R2_BUCKET_NAME "$DEV_VARS" | cut -d= -f2)
export AWS_ACCESS_KEY_ID="$AK" AWS_SECRET_ACCESS_KEY="$SK"
python3 - <<PY
import hashlib, hmac, urllib.request, urllib.parse, xml.etree.ElementTree as ET, os, datetime as dt, sys
prefix = """${PREFIX}"""
root   = """${ROOT}"""
acc    = """${ACC}"""
bucket = """${BUCKET}"""
host   = f"{acc}.r2.cloudflarestorage.com"

def sign(method, path, query, payload_hash, t):
    amz  = t.strftime("%Y%m%dT%H%M%SZ")
    date = t.strftime("%Y%m%d")
    qs   = "&".join(f"{k}={urllib.parse.quote(v, safe='')}" for k, v in sorted(query.items()))
    canon = f"{method}\n{path}\n{qs}\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz}\n\nhost;x-amz-content-sha256;x-amz-date\n{payload_hash}"
    scope = f"{date}/auto/s3/aws4_request"
    sts   = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(canon.encode()).hexdigest()}"
    def h(k, m): return hmac.new(k, m.encode(), hashlib.sha256).digest()
    sk = os.environ["AWS_SECRET_ACCESS_KEY"]
    kd  = h(f"AWS4{sk}".encode(), date); kr = h(kd, "auto"); ks = h(kr, "s3"); ki = h(ks, "aws4_request")
    sig = hmac.new(ki, sts.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization": f"AWS4-HMAC-SHA256 Credential={os.environ['AWS_ACCESS_KEY_ID']}/{scope}, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={sig}",
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz,
    }

def get(key, dst):
    t = dt.datetime.now(dt.timezone.utc)
    empty = hashlib.sha256(b"").hexdigest()
    headers = sign("GET", f"/{bucket}/{key}", {}, empty, t)
    url = f"https://{host}/{bucket}/{key}"
    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk: break
            f.write(chunk)

def list_prefix(prefix):
    out = []
    cont = None
    while True:
        t = dt.datetime.now(dt.timezone.utc)
        empty = hashlib.sha256(b"").hexdigest()
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if cont: q["continuation-token"] = cont
        headers = sign("GET", f"/{bucket}", q, empty, t)
        qs = "&".join(f"{k}={urllib.parse.quote(v, safe='')}" for k, v in sorted(q.items()))
        url = f"https://{host}/{bucket}?{qs}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            body = r.read()
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        rt = ET.fromstring(body)
        for c in rt.findall(f"{ns}Contents"):
            k = c.find(f"{ns}Key").text
            if not k.endswith("/"): out.append(k)
        tr = rt.find(f"{ns}IsTruncated")
        if tr is None or tr.text != "true": break
        nt = rt.find(f"{ns}NextContinuationToken")
        if nt is None: break
        cont = nt.text
    return out

# Single object?
single = not prefix.endswith("/")
if single:
    dst = os.path.join(root, prefix)
    print(f"GET {prefix} → {dst}")
    get(prefix, dst)
    sys.exit(0)

# Prefix
keys = list_prefix(prefix)
print(f"{len(keys)} objects under {prefix}")
for k in keys:
    dst = os.path.join(root, k)
    if os.path.exists(dst):
        print(f"  skip {k} (exists)")
        continue
    print(f"  GET {k}")
    get(k, dst)
print("done")
PY

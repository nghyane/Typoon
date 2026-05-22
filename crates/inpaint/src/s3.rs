// SPDX-License-Identifier: GPL-3.0-or-later
//! Tiny S3 client for R2 (no FUSE).
//!
//! Three ops: GET, PUT. Authenticated via AWS4-HMAC-SHA256 with R2
//! credentials. Reuses one `reqwest::Client` so HTTP/1.1 keep-alive
//! amortises TLS across the 4 R2 ops we do per page.
//!
//! Why not `aws-sdk-s3`? It pulls in ~80 deps and tokio-rustls features
//! we don't need. The signing routine here is ~120 LOC and covers exactly
//! what R2 expects (single-shot UNSIGNED-PAYLOAD writes, GET reads).

use std::time::SystemTime;

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

pub struct S3 {
    client:     reqwest::Client,
    endpoint:   String,            // https://{account}.r2.cloudflarestorage.com
    bucket:     String,
    region:     &'static str,      // "auto"
    access_key: String,
    secret_key: String,
}

impl S3 {
    pub fn from_env() -> Result<Self> {
        let account_id = std::env::var("R2_ACCOUNT_ID")
            .context("R2_ACCOUNT_ID env var missing")?;
        let bucket = std::env::var("R2_BUCKET_NAME")
            .context("R2_BUCKET_NAME env var missing")?;
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .context("AWS_ACCESS_KEY_ID env var missing")?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .context("AWS_SECRET_ACCESS_KEY env var missing")?;
        let endpoint = format!("https://{account_id}.r2.cloudflarestorage.com");
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(8)
            .build()
            .context("reqwest client init failed")?;
        Ok(Self { client, endpoint, bucket, region: "auto", access_key, secret_key })
    }

    pub async fn get(&self, key: &str) -> Result<Bytes> {
        let url = format!("{}/{}/{}", self.endpoint, self.bucket, key);
        let now = SystemTime::now();
        let payload_hash = sha256_hex(&[]);
        let auth = self.sign("GET", &url, &payload_hash, now)?;

        let resp = self.client
            .get(&url)
            .header("Host", self.host())
            .header("x-amz-date", amz_date(now))
            .header("x-amz-content-sha256", &payload_hash)
            .header("Authorization", auth)
            .send().await
            .with_context(|| format!("GET {key} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("GET {key} → {status}: {body}"));
        }
        Ok(resp.bytes().await?)
    }

    pub async fn put(&self, key: &str, body: Vec<u8>, content_type: &str) -> Result<()> {
        let url = format!("{}/{}/{}", self.endpoint, self.bucket, key);
        let now = SystemTime::now();
        let payload_hash = sha256_hex(&body);
        let mut auth_headers: Vec<(&str, String)> = vec![
            ("Content-Type",         content_type.to_string()),
            ("x-amz-content-sha256", payload_hash.clone()),
            ("x-amz-date",           amz_date(now)),
        ];
        let auth = self.sign_with_headers("PUT", &url, &payload_hash, now, &auth_headers)?;
        auth_headers.push(("Authorization", auth));

        let mut req = self.client.put(&url).header("Host", self.host()).body(body);
        for (k, v) in &auth_headers {
            req = req.header(*k, v);
        }
        let resp = req.send().await.with_context(|| format!("PUT {key} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("PUT {key} → {status}: {body}"));
        }
        Ok(())
    }

    fn host(&self) -> String {
        self.endpoint.trim_start_matches("https://").to_string()
    }

    fn sign(&self, method: &str, url: &str, payload_hash: &str, now: SystemTime) -> Result<String> {
        // GET: only Host + x-amz-date + x-amz-content-sha256 are signed.
        let headers: Vec<(&str, String)> = vec![
            ("x-amz-content-sha256", payload_hash.to_string()),
            ("x-amz-date",           amz_date(now)),
        ];
        self.sign_with_headers(method, url, payload_hash, now, &headers)
    }

    fn sign_with_headers(
        &self, method: &str, url: &str, payload_hash: &str,
        now: SystemTime, headers: &[(&str, String)],
    ) -> Result<String> {
        let parsed = url::Url::parse(url).context("bad URL")?;
        let canonical_uri = uri_encode_path(parsed.path());
        let canonical_qs  = canonical_query(parsed.query().unwrap_or(""));

        // Build canonical headers — sorted, lowercased keys, trimmed values.
        let mut hdrs: Vec<(String, String)> = headers.iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        hdrs.push(("host".into(), self.host()));
        hdrs.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical_headers = hdrs.iter()
            .map(|(k, v)| format!("{k}:{v}\n"))
            .collect::<String>();
        let signed_headers = hdrs.iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>().join(";");

        let canonical_request = format!(
            "{method}\n{canonical_uri}\n{canonical_qs}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        );
        let date_stamp = date_stamp(now);
        let amz_date   = amz_date(now);
        let scope = format!("{date_stamp}/{}/s3/aws4_request", self.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            sha256_hex(canonical_request.as_bytes()),
        );

        let k_date    = hmac(format!("AWS4{}", self.secret_key).as_bytes(), date_stamp.as_bytes());
        let k_region  = hmac(&k_date,   self.region.as_bytes());
        let k_service = hmac(&k_region, b"s3");
        let k_signing = hmac(&k_service, b"aws4_request");
        let signature = hex(hmac(&k_signing, string_to_sign.as_bytes()).as_slice());

        Ok(format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.access_key, scope, signed_headers, signature,
        ))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn sha256_hex(input: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(input);
    hex(&h.finalize())
}

fn hmac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac key");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        write!(s, "{:02x}", b).unwrap();
    }
    s
}

fn amz_date(now: SystemTime) -> String {
    let t = now.duration_since(SystemTime::UNIX_EPOCH).expect("clock");
    let secs = t.as_secs() as i64;
    format_timestamp(secs, true)
}

fn date_stamp(now: SystemTime) -> String {
    let t = now.duration_since(SystemTime::UNIX_EPOCH).expect("clock");
    let secs = t.as_secs() as i64;
    format_timestamp(secs, false)
}

// Inline UTC formatter — chrono pulls in too much for a single call site.
fn format_timestamp(unix_secs: i64, include_time: bool) -> String {
    let days = unix_secs.div_euclid(86_400);
    let rem  = unix_secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Gregorian date from days-since-1970-01-01 (Howard Hinnant algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y   = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp  = (5 * doy + 2) / 153;
    let d   = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m_  = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let y   = if m_ <= 2 { y + 1 } else { y };

    if include_time {
        format!("{:04}{:02}{:02}T{:02}{:02}{:02}Z", y, m_, d, h, m, s)
    } else {
        format!("{:04}{:02}{:02}", y, m_, d)
    }
}

fn uri_encode_path(path: &str) -> String {
    // Encode each segment, preserve slashes. R2 keys can contain colons +
    // slashes; the rest of allowed unreserved chars matches AWS rules.
    let mut out = String::with_capacity(path.len());
    for ch in path.chars() {
        if ch == '/' || ch.is_ascii_alphanumeric()
            || matches!(ch, '-' | '_' | '.' | '~') {
            out.push(ch);
        } else {
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

fn canonical_query(qs: &str) -> String {
    if qs.is_empty() { return String::new(); }
    let mut pairs: Vec<(String, String)> = qs.split('&')
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (uri_encode_component(k), uri_encode_component(v))
        })
        .collect();
    pairs.sort();
    pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

fn uri_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            out.push(ch);
        } else {
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

# Requirements Document

## Introduction

The web-svelte app (SvelteKit, currently deployed as a Discord Activity) integrates
manga sources through declarative adapters. The "comix" source (comix.to) is
currently broken and cannot be fixed from a web context.

Root cause (verified end-to-end during investigation):

1. comix.to sits behind a Cloudflare *managed challenge* ("Chờ một chút..." /
   "Just a moment..."). Plain `fetch`, TLS-fingerprint spoofing
   (`curl-impersonate`), and both headless and headful browser automation via CDP
   all remain stuck on the challenge indefinitely.
2. The comix request token is produced by a VM-obfuscated "secure" module whose
   decryption key is environment-locked and derived from a `cfg` token. The repo
   ships a stale `cfg` snapshot in `web-svelte/static/comix-vendor/manifest.json`
   (`fetchedAt: 2026-06-20`).
3. With a stale `cfg`, the VM fails with `VM decryption key not available`, so
   `secure.i(client)` cannot sign requests, axios never attaches `params._`, and
   the worker throws `no token`.

Reverse-engineering the token algorithm is not viable (encrypted VM bytecode,
rotating environment-locked key), and auto-refreshing `cfg` over the network is
blocked by Cloudflare. There is no browser extension available. The only reliable
way to use comix is from a real browser session that has passed the Cloudflare
challenge with trusted cookies.

This feature adds a **Tauri desktop target** (macOS/Windows/Linux) that runs
**alongside** the existing web build, sharing the same SvelteKit frontend. On
desktop, a native OS webview lets the user solve the Cloudflare challenge once;
the resulting trusted session and a freshly-read `cfg` are reused to generate
comix tokens, making the comix source work on desktop. The comix adapter routes
per-environment: on desktop it uses the native-webview path, and on web it is
unavailable. All other sources continue to work unchanged on both targets.
Android/mobile is out of scope.

## Glossary

- **Web_Target**: The existing SvelteKit build deployed as a Discord Activity / in a browser.
- **Desktop_Target**: The new Tauri desktop application (macOS/Windows/Linux) that wraps the same SvelteKit frontend.
- **Shared_Frontend**: The single SvelteKit codebase under `web-svelte/` consumed by both Web_Target and Desktop_Target.
- **Comix_Adapter**: The source adapter for comix.to located at `web-svelte/src/lib/source/adapters/comix.ts`.
- **Cloudflare_Challenge**: The Cloudflare managed challenge interstitial served by comix.to that must be solved by a real browser session.
- **Native_Webview**: An OS-provided webview (e.g. WKWebView / WebView2 / WebKitGTK) controlled by the Desktop_Target that navigates comix.to.
- **Cfg_Token**: The environment-locked `cfg` string required to decrypt and run the comix "secure" VM module that signs API requests.
- **Comix_Session**: The set of trusted cookies/state from a comix.to browsing session that has passed the Cloudflare_Challenge.
- **Environment_Detection**: The runtime check (e.g. presence of `window.__TAURI__`) that determines whether the Shared_Frontend is running under Desktop_Target or Web_Target.
- **Token_Bridge**: The Desktop_Target mechanism (Tauri command) that returns only comix token strings to the Shared_Frontend, keeping cookies/cfg isolated.

## Requirements

### Requirement 1: Tauri desktop target alongside web

**User Story:** As a maintainer, I want a Tauri desktop target that reuses the existing SvelteKit frontend, so that web and desktop run from one codebase without disrupting the web deployment.

#### Acceptance Criteria

1. THE Desktop_Target SHALL build from the same Shared_Frontend codebase used by the Web_Target.
2. WHEN the Web_Target is built and deployed, THE existing Discord Activity deployment SHALL continue to function unchanged.
3. THE Desktop_Target SHALL produce a runnable desktop application for at least macOS (the development platform).
4. THE Desktop_Target build configuration SHALL NOT require modifications to source-agnostic frontend code beyond Environment_Detection and Comix_Adapter routing.

### Requirement 2: Per-environment comix routing

**User Story:** As a user, I want comix to work when I run the desktop app and to be clearly handled when I use the web app, so that I am not shown a broken source.

#### Acceptance Criteria

1. WHEN the Shared_Frontend runs under the Desktop_Target, THE Comix_Adapter SHALL use the Native_Webview path to obtain tokens.
2. WHEN the Shared_Frontend runs under the Web_Target, THE Comix_Adapter SHALL be treated as unavailable.
3. WHEN comix is unavailable on the Web_Target, THE Shared_Frontend SHALL hide or disable the comix source in the source list so the user cannot select a non-functional source.
4. THE Environment_Detection SHALL determine the active target without requiring user configuration.

### Requirement 3: One-time Cloudflare challenge in native webview

**User Story:** As a desktop user, I want to solve the comix Cloudflare challenge once, so that subsequent comix requests work without repeated interruptions.

#### Acceptance Criteria

1. WHEN comix is first used on the Desktop_Target and no valid Comix_Session exists, THE Desktop_Target SHALL present the Native_Webview navigated to comix.to so the user can solve the Cloudflare_Challenge.
2. WHEN the user has solved the Cloudflare_Challenge, THE Desktop_Target SHALL persist the Comix_Session for reuse across requests.
3. WHILE a valid Comix_Session exists, THE Desktop_Target SHALL NOT require the user to solve the Cloudflare_Challenge again for routine comix requests.
4. WHEN a persisted Comix_Session is no longer accepted by comix.to, THE Desktop_Target SHALL re-present the Native_Webview to let the user solve the challenge again.

### Requirement 4: Fresh cfg acquisition

**User Story:** As a desktop user, I want comix tokens generated from a current cfg, so that the source does not break when comix.to rotates its cfg.

#### Acceptance Criteria

1. THE Desktop_Target SHALL read the Cfg_Token from the live, challenge-passed comix.to session rather than from a hardcoded snapshot.
2. WHEN comix.to serves an updated Cfg_Token, THE Desktop_Target SHALL use the updated value for subsequent token generation without a code change.
3. IF the Cfg_Token cannot be read from the current session, THEN THE Desktop_Target SHALL surface a recoverable error and prompt re-navigation through the Native_Webview rather than failing silently.

### Requirement 5: Token isolation

**User Story:** As a maintainer, I want comix cookies and cfg confined to the desktop webview boundary, so that only token strings cross back into the app.

#### Acceptance Criteria

1. THE Token_Bridge SHALL return only token strings (and necessary request parameters) to the Shared_Frontend.
2. THE Token_Bridge SHALL NOT expose raw Comix_Session cookies or the decrypted VM internals to the Shared_Frontend.
3. THE comix credential handling SHALL be confined to the Desktop_Target / Native_Webview boundary.

### Requirement 6: Other sources unaffected

**User Story:** As a user, I want all non-comix sources to keep working on both web and desktop, so that this change does not regress existing functionality.

#### Acceptance Criteria

1. THE non-comix source adapters (including otruyen, truyenqq, mangadex, baozimh, naver-webtoon) SHALL function unchanged on the Web_Target.
2. THE non-comix source adapters SHALL function unchanged on the Desktop_Target.
3. THE change SHALL NOT modify the public behavior of source-agnostic frontend code paths other than Environment_Detection and Comix_Adapter routing.

## Out of Scope

- Android / iOS / any mobile target.
- Reverse-engineering or replacing the comix token algorithm.
- Automating the Cloudflare challenge without user interaction.
- Changes to non-comix source adapters beyond verifying they remain unaffected.

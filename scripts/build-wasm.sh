#!/usr/bin/env bash
# Builds the WebAssembly module and generates its JavaScript bindings.
#
# The wasm-bindgen CLI version is pinned to match the crate exactly (=0.2.100). The two
# negotiate a schema over the generated binary, and a mismatch is a hard error, so a routine
# `cargo update` that bumps the crate will break this step until the CLI is bumped to match.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-wasm-release}"
OUT_DIR="web/src/wasm"

echo "building eskf-wasm ($PROFILE)"
cargo build -p eskf-wasm --target wasm32-unknown-unknown --profile "$PROFILE"

WASM="target/wasm32-unknown-unknown/$PROFILE/eskf_wasm.wasm"

echo "generating bindings"
# The loader imports the binary through Vite's ?url so the bundler emits and fingerprints the
# asset and passes the URL to init() explicitly, sidestepping the Vite 8 production-worker trap
# where the glue's own import.meta.url resolves to undefined.
wasm-bindgen "$WASM" --out-dir "$OUT_DIR" --target web

echo "wrote $OUT_DIR"
ls -la "$OUT_DIR"

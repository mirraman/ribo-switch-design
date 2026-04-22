#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pushd ribo_rs >/dev/null
maturin develop --release
popd >/dev/null

SRC=$(ls .venv/Lib/site-packages/ribo_rs/ribo_rs.cp*.pyd 2>/dev/null | head -1)
if [ -z "$SRC" ]; then
    SRC=$(ls .venv/lib/python*/site-packages/ribo_rs/ribo_rs*.so 2>/dev/null | head -1)
fi
if [ -z "$SRC" ]; then
    echo "error: could not find built extension in site-packages" >&2
    exit 1
fi

DEST="ribo_rs/$(basename "$SRC")"
cp "$SRC" "$DEST"
echo "synced: $SRC -> $DEST"

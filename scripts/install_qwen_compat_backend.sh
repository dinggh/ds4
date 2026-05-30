#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TAG=${LLAMA_CPP_RELEASE_TAG:-b9415}
OS=$(uname -s)
ARCH=$(uname -m)

case "$OS:$ARCH" in
  Darwin:arm64) ASSET="llama-${TAG}-bin-macos-arm64.tar.gz" ;;
  Darwin:x86_64) ASSET="llama-${TAG}-bin-macos-x64.tar.gz" ;;
  Linux:aarch64|Linux:arm64) ASSET="llama-${TAG}-bin-ubuntu-arm64.tar.gz" ;;
  Linux:x86_64) ASSET="llama-${TAG}-bin-ubuntu-x64.tar.gz" ;;
  *)
    echo "unsupported platform for prebuilt llama.cpp backend: $OS $ARCH" >&2
    exit 2
    ;;
esac

DEST="$ROOT/third_party/llama.cpp-bin"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ASSET}"
TMP="${TMPDIR:-/tmp}/${ASSET}"

mkdir -p "$DEST"
curl -L --fail -o "$TMP" "$URL"
tar -xzf "$TMP" -C "$DEST"

CLI=$(find "$DEST" -type f -name llama-cli | head -n 1)
if [ -z "$CLI" ]; then
  echo "llama-cli was not found after extracting $ASSET" >&2
  exit 1
fi
chmod +x "$CLI"
echo "$CLI"

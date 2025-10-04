#!/usr/bin/env bash
set -euo pipefail

# Create an anonymized submission archive excluding fluke-webpage/ and common noise

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/submission"
ARCHIVE="$OUT_DIR/fluke_submission.zip"

mkdir -p "$OUT_DIR"

echo "Building anonymized submission at $ARCHIVE"

# Use zip with explicit excludes (portable); alternative: git archive honors .gitattributes
cd "$ROOT_DIR"

EXCLUDES=(
  "fluke-webpage/*"
  "**/__pycache__/*"
  "**/*.pyc"
  "**/.DS_Store"
  "**/*.ipynb"
)

# Whitelist patterns to keep despite excludes
WHITELIST=(
  "fluke-source-code/data_generation/.*\\.ipynb$"
)

# Build file list safely
TMP_LIST="$OUT_DIR/filelist.txt"
rm -f "$TMP_LIST"
git ls-files > "$TMP_LIST"

# Filter excludes while honoring whitelist
for pattern in "${EXCLUDES[@]}"; do
  # Convert glob to grep -vE pattern by escaping * to .*
  re=$(printf '%s\n' "$pattern" | sed -E 's/[].[^$\{}|()+?]/\\&/g; s/\*/.*/g')
  tmp="$TMP_LIST.tmp"
  # Keep whitelisted files
  if [ ${#WHITELIST[@]} -gt 0 ]; then
    # Build combined whitelist regex
    wl=$(printf '%s\n' "${WHITELIST[@]}" | paste -sd '|' -)
    awk -v ex="$re" -v wl="$wl" '
      $0 ~ wl { print; next }  # keep whitelisted
      $0 ~ "^" ex "$" { next } # drop excluded match
      { print }
    ' "$TMP_LIST" > "$tmp" || true
  else
    grep -vE "^$re$" "$TMP_LIST" > "$tmp" || true
  fi
  mv "$tmp" "$TMP_LIST"
done

# Create zip
rm -f "$ARCHIVE"
zip -q -@ "$ARCHIVE" < "$TMP_LIST"

echo "Done: $ARCHIVE"

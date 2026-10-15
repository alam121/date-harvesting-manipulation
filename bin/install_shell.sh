#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASHRC="${HOME}/.bashrc"

BEGIN_MARKER="# >>> manipulatorsdatepalm commands >>>"
END_MARKER="# <<< manipulatorsdatepalm commands <<<"

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

if [[ -f "$BASHRC" ]]; then
    awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
        $0 == begin {skip=1; next}
        $0 == end {skip=0; next}
        !skip {print}
    ' "$BASHRC" > "$TMP_FILE"
else
    : > "$TMP_FILE"
fi

cat >> "$TMP_FILE" <<EOF
$BEGIN_MARKER
export MANIPULATOR_REPO="$REPO_ROOT"
PATH=":\$PATH:"
PATH="\${PATH//:\$MANIPULATOR_REPO\\/bin:/:}"
PATH="\${PATH#:}"
PATH="\${PATH%:}"
export PATH="\$MANIPULATOR_REPO/bin:\$PATH"
$END_MARKER
EOF

mv "$TMP_FILE" "$BASHRC"
echo "Updated $BASHRC"
echo "Run: source ~/.bashrc"

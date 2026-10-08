#!/usr/bin/env bash
# Install/remove the Rabbit Code CLI helper from a trusted source checkout.
# Never downloads code, edits shell startup files, or needs sudo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${RABBIT_BIN_DIR:-$HOME/.local/bin}"
LAUNCHER="$BIN_DIR/rabbit"
MARKER="# RABBIT-CODE-MANAGED-CLI v1"
MODE="${1:---install}"

case "$MODE" in
  --install|--update|--uninstall) ;;
  *)
    echo "Usage: bash scripts/install_rabbit.sh [--install|--update|--uninstall]" >&2
    exit 2
    ;;
esac

if [[ -e "$LAUNCHER" || -L "$LAUNCHER" ]]; then
  if [[ ! -f "$LAUNCHER" || -L "$LAUNCHER" ]] || \
     ! grep -Fqx "$MARKER" "$LAUNCHER"; then
    echo "Refusing to replace an existing unmanaged command: $LAUNCHER" >&2
    exit 3
  fi
fi

if [[ "$MODE" == "--uninstall" ]]; then
  if [[ -f "$LAUNCHER" ]]; then
    rm -- "$LAUNCHER"
    echo "Removed Rabbit Code helper: $LAUNCHER"
  else
    echo "Rabbit Code helper is not installed at: $LAUNCHER"
  fi
  exit 0
fi

if [[ ! -f "$REPO_ROOT/src/rabbit_foundry/helper_cli.py" ]]; then
  echo "Source checkout is missing Rabbit Code helper files." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.11+ with your distribution's package manager." >&2
  exit 2
fi
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Rabbit Code requires Python 3.11+." >&2
  exit 2
fi

mkdir -p -- "$BIN_DIR"
printf -v src_path '%q' "$REPO_ROOT/src"
tmp_file="$(mktemp "$BIN_DIR/.rabbit-helper.XXXXXXXX")"
trap 'rm -f -- "$tmp_file"' EXIT

cat > "$tmp_file" <<EOF
#!/usr/bin/env bash
$MARKER
set -euo pipefail
export PYTHONPATH=$src_path\${PYTHONPATH:+:\$PYTHONPATH}
exec python3 -m rabbit_foundry.helper_cli "\$@"
EOF

chmod 755 "$tmp_file"
mv -f -- "$tmp_file" "$LAUNCHER"
trap - EXIT
echo "Rabbit Code helper installed: $LAUNCHER"
echo "This launcher uses your existing checkout: $REPO_ROOT"
echo "No dependency packages, downloads, or system-level changes were made."

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo "To use rabbit in this Bash session:"
    printf '  export PATH=%q:"$PATH"\n' "$BIN_DIR"
    echo "To use it in new Bash sessions, add that export line to ~/.bashrc."
    ;;
esac

echo "Next: rabbit doctor   |   rabbit models   |   rabbit ui"

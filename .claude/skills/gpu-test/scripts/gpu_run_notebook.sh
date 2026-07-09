#!/usr/bin/env bash
# Execute a dimma example notebook end-to-end on the remote CUDA GPU box.
#
# Same push -> ssh -> cd -> pull loop as gpu_test.sh, but instead of pytest it
# runs the notebook headless with `jupyter nbconvert --execute`. A non-zero exit
# means a cell raised; nbconvert prints the failing cell's traceback.
#
# Usage:
#   gpu_run_notebook.sh <notebook-path-relative-to-repo> [-- <extra nbconvert args>]
#
# Env overrides (same semantics as gpu_test.sh):
#   REMOTE_DIR  BRANCH  ALLOW_DIRTY  NO_PUSH  SKIP_DEPS
#   NB_TIMEOUT  per-cell timeout in seconds (default: 1200)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_RUN="$SCRIPT_DIR/ssh_run.sh"
LOCAL_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

NB="${1:?usage: gpu_run_notebook.sh <notebook-path> [-- <nbconvert args>]}"; shift
REMOTE_DIR="${REMOTE_DIR:-/home/diegol/dimma}"
BRANCH="${BRANCH:-$(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD)}"
NB_TIMEOUT="${NB_TIMEOUT:-1200}"

[ -f "$LOCAL_REPO/$NB" ] || { echo "notebook not found locally: $NB" >&2; exit 66; }

NBCONVERT_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; NBCONVERT_ARGS=("$@"); break ;;
    *)  echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---- Step 1: push local branch -----------------------------------------------
say "Step 1/5  push  (branch: $BRANCH)"
if [ -n "$(git -C "$LOCAL_REPO" status --porcelain)" ]; then
  if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
    echo "WARNING: uncommitted local changes will NOT be run (only pushed commits are)."
  else
    echo "ABORT: working tree is dirty. Commit your changes first, or set ALLOW_DIRTY=1." >&2
    git -C "$LOCAL_REPO" status --short >&2
    exit 1
  fi
fi
if [ "${NO_PUSH:-0}" = "1" ]; then
  echo "NO_PUSH=1 -> skipping push."
else
  git -C "$LOCAL_REPO" push -u origin "$BRANCH"
fi

# ---- Steps 2-5: ssh + cd + pull + deps + execute notebook on GPU -------------
DEPS_CMD='.venv/bin/python -m pip install -q -e ".[examples]"'
[ "${SKIP_DEPS:-0}" = "1" ] && DEPS_CMD='echo "SKIP_DEPS=1 -> skipping dep install"'

# Execute in place to a throwaway output file so the tracked notebook stays clean.
OUT_NB="/tmp/executed_$(basename "$NB")"

REMOTE_SCRIPT=$(cat <<REMOTE
set -e
cd "$REMOTE_DIR"
echo "== Step 3/5  cd == \$(pwd)"

echo "== Step 4/5  pull  (branch: $BRANCH) =="
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
git log -1 --oneline

echo "== deps ([examples] for jupyter/nbconvert) =="
$DEPS_CMD

echo "== GPU visibility =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

echo "== Step 5/5  execute notebook on GPU: $NB =="
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=$NB_TIMEOUT \
  --output "$OUT_NB" \
  "$NB" ${NBCONVERT_ARGS[*]:-}
echo "NOTEBOOK_OK"
REMOTE
)

say "Steps 2-5  ssh -> pull -> deps -> execute notebook on GPU  (remote: $REMOTE_DIR)"
"$SSH_RUN" "$REMOTE_SCRIPT"

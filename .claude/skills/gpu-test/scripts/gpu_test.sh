#!/usr/bin/env bash
# Run dimma's test suite on the remote CUDA GPU box.
#
# Five steps: push local branch -> ssh in -> cd repo -> pull -> pytest on GPU.
# The remote GPU is an NVIDIA RTX 6000 Ada (CUDA), which the Mac (Metal/CPU) can't provide.
#
# Usage:
#   gpu_test.sh [-- <extra pytest args>]
#
# Env overrides:
#   REMOTE_DIR   remote repo path        (default: /home/diegol/dimma)
#   BRANCH       branch to test          (default: current local branch)
#   ALLOW_DIRTY  =1 to test even with uncommitted local changes (they are NOT pushed)
#   NO_PUSH      =1 to skip the local push (assume remote already has the commits)
#   SKIP_DEPS    =1 to skip the idempotent `pip install -e .[dev]` step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_RUN="$SCRIPT_DIR/ssh_run.sh"
LOCAL_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

REMOTE_DIR="${REMOTE_DIR:-/home/diegol/dimma}"
BRANCH="${BRANCH:-$(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD)}"

# Everything after `--` is forwarded to pytest.
PYTEST_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; PYTEST_ARGS=("$@"); break ;;
    *)  echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---- Step 1: push local branch -----------------------------------------------
say "Step 1/5  push  (branch: $BRANCH)"
if [ -n "$(git -C "$LOCAL_REPO" status --porcelain)" ]; then
  if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
    echo "WARNING: uncommitted local changes will NOT be tested (only pushed commits are)."
  else
    echo "ABORT: working tree is dirty. Commit your changes first, or set ALLOW_DIRTY=1." >&2
    git -C "$LOCAL_REPO" status --short >&2
    exit 1
  fi
fi
if [ "${NO_PUSH:-0}" = "1" ]; then
  echo "NO_PUSH=1 -> skipping push."
else
  git -C "$LOCAL_REPO" push origin "$BRANCH"
fi

# ---- Steps 2-5: ssh + cd + pull + (deps) + pytest on GPU ---------------------
DEPS_CMD='echo "SKIP_DEPS=1 -> skipping dep install"'
[ "${SKIP_DEPS:-0}" = "1" ] || DEPS_CMD='.venv/bin/python -m pip install -q -e ".[dev]"'

# Build the remote script. Fail fast at each stage; confirm the CUDA device before running.
REMOTE_SCRIPT=$(cat <<REMOTE
set -e
cd "$REMOTE_DIR"
echo "== Step 3/5  cd == \$(pwd)"

echo "== Step 4/5  pull  (branch: $BRANCH) =="
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
git log -1 --oneline

echo "== deps =="
$DEPS_CMD

echo "== GPU visibility =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

echo "== Step 5/5  pytest on GPU =="
.venv/bin/python -m pytest ${PYTEST_ARGS[*]:-}
REMOTE
)

say "Steps 2-5  ssh -> pull -> deps -> pytest on GPU  (remote: $REMOTE_DIR)"
"$SSH_RUN" "$REMOTE_SCRIPT"

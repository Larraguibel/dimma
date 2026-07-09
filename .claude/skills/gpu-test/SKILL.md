---
name: gpu-test
description: Run dimma's test suite (or any pytest) on the remote NVIDIA CUDA GPU box that the Mac cannot provide. Pushes the local branch, SSHes into the Linux server, pulls, and runs pytest on the GPU. Use when the user wants to test CUDA-GPU functionality, run tests on the GPU, verify JAX-on-CUDA behavior, or says "test on the GPU / server / remote".
---

# gpu-test

Runs the 5-step loop **push → ssh → cd → pull → pytest-on-GPU** against the FONDECYT GPU
server (NVIDIA RTX 6000 Ada, CUDA). The Mac only has Metal/CPU, so CUDA-specific
behavior must be verified here.

## Prerequisites (already true in this repo)

- `.env` at repo root holds `SSH_USER`, `SHH_PWD`, `SSH_PORT`, `SHH_IP` (note the `SHH` typos are intentional — the scripts read these exact names). `.env` is gitignored; never commit or print it.
- Local machine has `expect` (ships with macOS) — used to feed the SSH password without echoing it.
- Server has the repo at `/home/diegol/dimma` on a repo-local `.venv`, JAX+CUDA12, and a GitHub deploy key authorized for `git pull`.

## Quick start

Run the whole loop (full suite, current branch):

```bash
.claude/skills/gpu-test/scripts/gpu_test.sh
```

Run a subset by forwarding args to pytest (everything after `--`):

```bash
.claude/skills/gpu-test/scripts/gpu_test.sh -- tests/test_device.py -v
```

Ad-hoc remote command (no push/test) via the raw SSH helper:

```bash
.claude/skills/gpu-test/scripts/ssh_run.sh 'nvidia-smi'
```

## What gpu_test.sh does

1. **Push** — aborts if the working tree is dirty (uncommitted changes are *not* tested); pushes the current branch to `origin`.
2–4. **SSH → cd → pull** — `cd /home/diegol/dimma`, `git fetch`, checkout the same branch, `git pull --ff-only`.
5. **pytest on GPU** — idempotently installs dev extras (`pip install -e .[dev]`, needed because the venv ships without `pytest`), prints `nvidia-smi` + `jax.devices()` to confirm the CUDA device, then runs `pytest`.

## Options (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `BRANCH` | current local branch | branch to push/test |
| `REMOTE_DIR` | `/home/diegol/dimma` | remote repo path |
| `ALLOW_DIRTY` | `0` | `1` = run despite uncommitted local changes (they won't be pushed) |
| `NO_PUSH` | `0` | `1` = skip push (remote already has the commits) |
| `SKIP_DEPS` | `0` | `1` = skip the dev-extras install (after the first successful run) |

## Notes

- The password is fed via `expect` with output suppressed; it must never appear in logs. Values are auto-trimmed for stray whitespace / non-breaking spaces.
- Commit your work before running — only pushed commits reach the GPU box.
- If `git pull` fails on auth, verify the box's key: `ssh_run.sh 'ssh -T git@github.com'`.

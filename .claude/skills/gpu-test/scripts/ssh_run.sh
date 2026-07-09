#!/usr/bin/env bash
# Non-interactive SSH runner for the dimma CUDA-GPU box.
#
# Reads credentials from the repo .env, feeds the password via `expect`,
# and NEVER prints the password (log_user 0 while sending it).
# Trims stray whitespace / UTF-8 non-breaking spaces (U+00A0) that sneak
# into hand-edited .env values.
#
# Usage: ssh_run.sh "<remote command>"
# Env overrides: DIMMA_ENV (path to .env)
set -euo pipefail

# Resolve .env: explicit override, else repo root two levels above this skill.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV="$(cd "$SCRIPT_DIR/../../../.." && pwd)/.env"
ENV_FILE="${DIMMA_ENV:-$DEFAULT_ENV}"

[ -f "$ENV_FILE" ] || { echo "ssh_run: .env not found at $ENV_FILE" >&2; exit 4; }
set -a; . "$ENV_FILE"; set +a

: "${SSH_USER:?missing SSH_USER in .env}"
: "${SHH_PWD:?missing SHH_PWD in .env}"
: "${SSH_PORT:?missing SSH_PORT in .env}"
: "${SHH_IP:?missing SHH_IP in .env}"

# Strip leading/trailing ASCII whitespace, CR, and UTF-8 non-breaking space (U+00A0).
trim() { printf '%s' "$1" | LC_ALL=C sed -E 's/^([[:space:]]|\xc2\xa0)+//; s/([[:space:]]|\xc2\xa0)+$//'; }
SSH_USER="$(trim "$SSH_USER")"
SHH_PWD="$(trim "$SHH_PWD")"
SSH_PORT="$(trim "$SSH_PORT")"
SHH_IP="$(trim "$SHH_IP")"

REMOTE_CMD="${1:?usage: ssh_run.sh \"<remote command>\"}"
export SSHPW="$SHH_PWD"
export SSH_USER SSH_PORT SHH_IP REMOTE_CMD

expect <<'EOF'
set timeout 900
set user $env(SSH_USER)
set ip   $env(SHH_IP)
set port $env(SSH_PORT)
set pw   $env(SSHPW)
set cmd  $env(REMOTE_CMD)

spawn ssh -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $user@$ip $cmd
expect {
    -re {(?i)password:} {
        log_user 0
        send "$pw\r"
        log_user 1
        exp_continue
    }
    -re {(?i)permission denied} { puts "\nAUTH_FAILED"; exit 2 }
    timeout { puts "\nTIMEOUT"; exit 3 }
    eof
}
catch wait result
exit [lindex $result 3]
EOF

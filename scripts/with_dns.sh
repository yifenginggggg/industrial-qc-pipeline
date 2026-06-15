#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/with_dns.sh <command ...>" >&2
  exit 1
fi

TMP_RESOLV="$(mktemp)"
trap 'rm -f "${TMP_RESOLV}"' EXIT

cat > "${TMP_RESOLV}" <<'EOF'
nameserver 8.8.8.8
nameserver 1.1.1.1
options timeout:1 attempts:1
EOF

WORKDIR="$(pwd)"
COMMAND="$(printf '%q ' "$@")"

unshare --user --map-root-user --mount bash -lc \
  "mount --bind '${TMP_RESOLV}' /etc/resolv.conf && cd '${WORKDIR}' && ${COMMAND}"

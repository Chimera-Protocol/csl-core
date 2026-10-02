#!/usr/bin/env bash
# Builds the demo used by the README recordings in /tmp/csl-demo (a fixture host, nothing real).
# Run from the repository root:  bash docs/assets/tapes/prepare.sh
set -euo pipefail
REPO="$(pwd)"
D=/tmp/csl-demo
rm -rf "$D"
mkdir -p "$D/ws" "$D/mapping"
cp -R "$REPO/tests/venom/fixtures/host_ops" "$D/host"
cp "$REPO/tests/venom/fixtures/mapping_bypass/"{ops.csl,mapper_naive.py,mapper_fixed.py} "$D/mapping/"
# a finished setup for the studio and watch recordings
(cd "$D/ws" && cslcore setup --root ../host --yes --activate >/dev/null)
python "$REPO/scripts/venom_demo_traffic.py" --workspace "$D/ws" --count 600 --seed 7 >/dev/null
# the mapping test of a hand-written mapper (docs/assets/mapping.png)
cat > "$D/mapping/check.sh" <<'SH'
cslcore map --agent claude-code:ops --policy ../mapping/ops.csl --mapping ../mapping/mapper_naive.py:classify --test \
  --classify path_ok=scope:file_path --classify cmd_ok=command:command --classify dest_ok=destination:url \
  --allowed-root /srv/app --allowed-command "git status" --allowed-destination https://api.example.com/v1
SH
echo "demo ready in $D"

#!/usr/bin/env bash
set -euo pipefail

: "${MASK_CAMPAIGN_RUN_ID:?MASK_CAMPAIGN_RUN_ID is required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"

campaign_root=/workspace/qwen38-campaign/mask-campaign/v2
packet_root="$campaign_root/packet"
project_root="$packet_root/code/qwen3-vl-seg-decoder"
run_root="/workspace/qwen38-campaign/mask-campaign/preflight-runs/$MASK_CAMPAIGN_RUN_ID"
control_path="$campaign_root/control/$MASK_CAMPAIGN_RUN_ID.json"
guard_heartbeat="$run_root/pod-guard-heartbeat.json"
mask_venv="$campaign_root/runtime/venv-mask-v2"
sentinel="$campaign_root/MIG_PREFLIGHT_COMPLETE.json"
guard_pid=

mkdir -p "$run_root"
exec > >(tee -a "$run_root/preflight.log") 2>&1

mark_fatal() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 && ! -f "$run_root/RUN_COMPLETE" ]]; then
    temporary="$run_root/.RUN_FATAL.$$.tmp"
    printf 'MIG preflight failed with status %s\n' "$status" > "$temporary"
    mv "$temporary" "$run_root/RUN_FATAL"
    if [[ -n "$guard_pid" ]] && kill -0 "$guard_pid" 2>/dev/null; then
      wait "$guard_pid" || true
    fi
  fi
  return "$status"
}
trap mark_fatal EXIT

for _attempt in $(seq 1 120); do
  [[ -f "$control_path" ]] && break
  sleep 1
done
if [[ ! -f "$control_path" ]]; then
  echo "MIG launch control did not arrive within 120 seconds" >&2
  exit 70
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  timeout 300 apt-get update
  timeout 300 apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv ca-certificates git
  rm -rf /var/lib/apt/lists/*
fi

PYTHONPATH="$project_root/src" python3.12 -m ownership_decoder.remote_pod_guard \
  --control "$control_path" \
  --run-root "$run_root" \
  --heartbeat "$guard_heartbeat" \
  --interval-seconds 30 &
guard_pid=$!

# The manifest hash is bound by the API-side launch control. Before executing
# any packet-provided bootstrap or Python module, verify the complete frozen
# runtime and environment inventories using only the system standard library.
verify_critical_packet_files() {
  python3.12 - "$packet_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for field in ("runtime_files", "environment_files"):
    records = manifest.get(field)
    if not isinstance(records, list) or not records:
        raise SystemExit(f"packet {field} inventory is absent")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise SystemExit(f"packet {field} record schema is invalid")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe packet path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f"packet file is absent: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != int(record["size_bytes"]) or digest != record["sha256"]:
            raise SystemExit(f"packet file integrity failure: {relative}")
PY
}

packet_sha256=$(
  PYTHONPATH="$project_root/src" python3.12 - "$control_path" "$packet_root/manifest.json" "$MASK_CAMPAIGN_RUN_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from ownership_decoder.remote_pod_guard import PodGuardContract

control = PodGuardContract.from_mapping(
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
)
if control.run_id != sys.argv[3]:
    raise SystemExit("MIG launch control run ID mismatch")
digest = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if digest != control.packet_sha256:
    raise SystemExit("MIG launch packet SHA-256 mismatch")
print(digest)
PY
)

verify_critical_packet_files

bash "$packet_root/environment/bootstrap-mask-env.sh" \
  "$mask_venv" \
  "$campaign_root/repos/sam3-official"

"$mask_venv/bin/python" "$project_root/scripts/verify_mask_packet.py" \
  --packet-root "$packet_root" \
  --input-root /workspace/qwen38-campaign/launch-packet/inputs/breadth \
  --output "$run_root/packet-verification.json"

"$mask_venv/bin/python" "$project_root/scripts/run_mask_mig_preflight.py" \
  --packet-manifest "$packet_root/manifest.json" \
  --packet-sha256 "$packet_sha256" \
  --sam-repo "$campaign_root/repos/sam3-official" \
  --sam-revision 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da \
  --sam31-checkpoint "$campaign_root/models/sam3.1/sam3.1_multiplex.pt" \
  --sam31-checkpoint-sha256 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 \
  --sam31-checkpoint-size-bytes 3502755717 \
  --sam3-model-directory "$campaign_root/models/sam3" \
  --sam3-artifact-manifest "$packet_root/models/sam3-model-artifacts.json" \
  --workspace /workspace \
  --output "$sentinel"

python3.12 - "$sentinel" "$run_root/RUN_COMPLETE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
payload = {
    "format": "ownership-mask-mig-run-complete-v1",
    "sentinel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

wait "$guard_pid" || true

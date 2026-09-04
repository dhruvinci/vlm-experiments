#!/usr/bin/env bash
set -euo pipefail

: "${MASK_CAMPAIGN_RUN_ID:?MASK_CAMPAIGN_RUN_ID is required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"

campaign_root=/workspace/qwen38-campaign/mask-campaign/v2
packet_root="$campaign_root/packet"
project_root="$packet_root/code/qwen3-vl-seg-decoder"
run_root="/workspace/qwen38-campaign/mask-campaign/runs/$MASK_CAMPAIGN_RUN_ID"
control_path="$campaign_root/control/$MASK_CAMPAIGN_RUN_ID.json"
guard_heartbeat="$run_root/pod-guard-heartbeat.json"
mask_venv="$campaign_root/runtime/venv-mask-v2"
input_root=/workspace/qwen38-campaign/launch-packet/inputs/breadth
guard_pid=

mkdir -p "$run_root"
exec > >(tee -a "$run_root/launch.log") 2>&1

mark_fatal() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 && ! -f "$run_root/RUN_COMPLETE" ]]; then
    temporary="$run_root/.RUN_FATAL.$$.tmp"
    printf 'launch failed with status %s\n' "$status" > "$temporary"
    mv "$temporary" "$run_root/RUN_FATAL"
    # Keep the container entrypoint alive until the independent guard has seen
    # the fatal sentinel and the RunPod DELETE request has succeeded. The guard
    # itself retries transient API failures indefinitely.
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
  echo "launch control did not arrive within 120 seconds" >&2
  exit 70
fi

# The pinned NVIDIA CUDA image is deliberately minimal. Restore only the
# system interpreter required by the persistent, packet-pinned virtualenv.
# Both package-manager operations are time-bounded; the external controller
# covers this short interval before the in-Pod guard can start.
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

# Verify all executable/runtime files against the API-bound manifest before
# running packet-supplied bootstrap code. This intentionally uses only the
# system standard library; the exhaustive data/config verifier runs afterward.
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

mapfile -t control_fields < <(
  python3.12 - "$control_path" "$packet_root/manifest.json" "$MASK_CAMPAIGN_RUN_ID" "$campaign_root/MIG_PREFLIGHT_COMPLETE.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

control_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_run_id = sys.argv[3]
control = json.loads(control_path.read_text(encoding="utf-8"))
expected_keys = {
    "pod_id",
    "run_id",
    "billing_started_at",
    "spend_before_usd",
    "frozen_hourly_rate_usd",
    "terminate_at_usd",
    "max_runtime_seconds",
    "packet_sha256",
}
if set(control) != expected_keys:
    raise SystemExit("launch control keys are invalid")
if control["run_id"] != expected_run_id:
    raise SystemExit("launch control run ID mismatch")
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if control["packet_sha256"] != digest:
    raise SystemExit("launch control packet SHA-256 mismatch")
preflight_path = Path(sys.argv[4])
try:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit("validated MIG preflight sentinel is absent") from error
if preflight.get("format") != "ownership-mask-mig-preflight-v1":
    raise SystemExit("MIG preflight sentinel format mismatch")
if preflight.get("packet_sha256") != digest:
    raise SystemExit("MIG preflight was not run against this packet")
if preflight.get("sam_repo_revision") != "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da":
    raise SystemExit("MIG preflight SAM revision mismatch")
if preflight.get("artifact_count") != 9 or preflight.get("model_weights_loaded") is not False:
    raise SystemExit("MIG preflight artifact or model-load contract mismatch")
if preflight.get("runtime_smoke", {}).get("finite") is not True:
    raise SystemExit("MIG preflight runtime smoke did not pass")
for key in (
    "pod_id",
    "spend_before_usd",
    "frozen_hourly_rate_usd",
    "terminate_at_usd",
    "billing_started_at",
    "max_runtime_seconds",
):
    value = str(control[key])
    if "\n" in value or "\r" in value:
        raise SystemExit(f"launch control field contains a newline: {key}")
    print(value)
PY
)

if [[ ${#control_fields[@]} -ne 6 ]]; then
  echo "launch control did not produce six validated fields" >&2
  exit 70
fi
pod_id=${control_fields[0]}
spend_before_usd=${control_fields[1]}
hourly_rate_usd=${control_fields[2]}
terminate_at_usd=${control_fields[3]}
billing_started_at=${control_fields[4]}
max_runtime_seconds=${control_fields[5]}

verify_critical_packet_files

bash "$packet_root/environment/bootstrap-mask-env.sh" \
  "$mask_venv" \
  "$campaign_root/repos/sam3-official"

"$mask_venv/bin/python" "$project_root/scripts/verify_mask_packet.py" \
  --packet-root "$packet_root" \
  --input-root "$input_root" \
  --output "$run_root/packet-verification.json"

"$mask_venv/bin/python" "$project_root/scripts/supervise_sam31_breadth.py" \
  --output-root "$run_root" \
  --max-restarts 2 \
  --poll-seconds 30 \
  --max-runtime-seconds "$max_runtime_seconds" \
  --terminate-grace-seconds 20 \
  --pod-id "$pod_id" \
  --spend-before-usd "$spend_before_usd" \
  --hourly-rate-usd "$hourly_rate_usd" \
  --terminate-at-usd "$terminate_at_usd" \
  --billing-started-at "$billing_started_at" \
  -- \
  "$mask_venv/bin/python" "$project_root/scripts/run_remote_mask_campaign.py" \
  --config "$project_root/configs/breadth-tracking/back_seatbelt.json" \
  --config "$project_root/configs/breadth-tracking/guard_scramble.json" \
  --config "$project_root/configs/breadth-tracking/half_guard.json" \
  --config "$project_root/configs/breadth-tracking/mount.json" \
  --input-root "$input_root" \
  --output-root "$run_root" \
  --sam-repo "$campaign_root/repos/sam3-official" \
  --sam-revision 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da \
  --sam31-checkpoint "$campaign_root/models/sam3.1/sam3.1_multiplex.pt" \
  --sam31-checkpoint-sha256 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 \
  --sam3-model-directory "$campaign_root/models/sam3" \
  --sam3-artifact-manifest "$packet_root/models/sam3-model-artifacts.json" \
  --workspace /workspace \
  --minimum-prompt-area 64 \
  --box-padding-fraction 0.01

wait "$guard_pid" || true

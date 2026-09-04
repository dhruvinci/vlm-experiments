#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VENV_PATH SAM_REPO_PATH" >&2
  exit 64
fi

mask_venv_path=$1
sam_repo_path=$2
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bootstrap_marker="$mask_venv_path/.ownership-mask-runtime-ready"

if [[ ! -d "$sam_repo_path/.git" ]]; then
  echo "SAM repository is not a Git checkout: $sam_repo_path" >&2
  exit 65
fi

sam_revision=$(git -C "$sam_repo_path" rev-parse HEAD)
if [[ "$sam_revision" != "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da" ]]; then
  echo "SAM repository revision mismatch: $sam_revision" >&2
  exit 66
fi

cu_requirements_sha=$(sha256sum "$script_directory/requirements-cu130.txt" | cut -d' ' -f1)
mask_requirements_sha=$(sha256sum "$script_directory/requirements-mask.txt" | cut -d' ' -f1)
expected_marker="sam_revision=$sam_revision cu_requirements=$cu_requirements_sha mask_requirements=$mask_requirements_sha"

if [[ -f "$bootstrap_marker" ]]; then
  if [[ "$(<"$bootstrap_marker")" != "$expected_marker" ]]; then
    echo "existing mask runtime marker does not match the frozen environment" >&2
    exit 67
  fi
  "$mask_venv_path/bin/python" -c 'import importlib.metadata, numpy, torch, torchvision, transformers, sam3; assert numpy.__version__ == "1.26.4"; assert torch.version.cuda == "13.0"; assert importlib.metadata.version("sam3") == "0.1.0"'
  exit 0
fi

if [[ -e "$mask_venv_path" ]]; then
  echo "refusing to modify an unverified existing environment: $mask_venv_path" >&2
  exit 68
fi

python3.12 -m venv "$mask_venv_path"
"$mask_venv_path/bin/python" -m pip install --upgrade pip==25.3
"$mask_venv_path/bin/python" -m pip install --no-cache-dir -r "$script_directory/requirements-cu130.txt"
"$mask_venv_path/bin/python" -m pip install --no-cache-dir -r "$script_directory/requirements-mask.txt"
"$mask_venv_path/bin/python" -m pip install --no-deps -e "$sam_repo_path"

"$mask_venv_path/bin/python" -c 'import importlib.metadata, numpy, torch, torchvision, transformers, sam3; assert numpy.__version__ == "1.26.4"; assert torch.__version__ == "2.12.1+cu130"; assert torchvision.__version__ == "0.27.1+cu130"; assert torch.version.cuda == "13.0"; assert transformers.__version__ == "5.16.1"; assert importlib.metadata.version("sam3") == "0.1.0"'
printf '%s\n' "$expected_marker" > "$bootstrap_marker"

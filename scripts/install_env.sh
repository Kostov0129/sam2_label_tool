#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-probe_sam2}"
SAM2_DIR="${SAM2_DIR:-third_party/sam2}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

conda env create -f environment.yml -n "${ENV_NAME}" || conda env update -f environment.yml -n "${ENV_NAME}"

mkdir -p third_party
if [ ! -d "${SAM2_DIR}/.git" ]; then
  git clone https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
fi

conda run -n "${ENV_NAME}" python -m pip install -e "${SAM2_DIR}"

echo "Done. Activate with: conda activate ${ENV_NAME}"

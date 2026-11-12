#!/usr/bin/env bash
# Local debug set: 3 tasks (one per main repo), sandbox wheels, and the harness Wheelhouse.
# Run inside WSL from the repo root:  bash kaggle_swe/download_debug_set.sh
set -eo pipefail
source scripts/env.sh
python kaggle_swe/download.py tasks fastapi_14492 rich_3278 requests_6589
python kaggle_swe/download.py support
python kaggle_swe/download.py wheelhouse
du -sh ~/data/gemma4/* 2>/dev/null
echo "debug set done"

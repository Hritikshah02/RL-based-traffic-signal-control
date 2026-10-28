#!/usr/bin/env bash
# Train on Kaggle and pull the models back into models/.
#
#   kaggle/run.sh [methods] [hours]      e.g.  kaggle/run.sh all       or  kaggle/run.sh FRAP,PPO 300
#
# TSC_COPY=0 keeps the downloaded models in kaggle/out/result/models instead of overwriting models/.
#
# Needs: Kaggle token in ~/.kaggle (access_token or kaggle.json), a recent `kaggle` CLI
# (set KAGGLE=/path/to/kaggle to choose one), and the branch pushed to GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."

KAGGLE=${KAGGLE:-kaggle}
METHODS=${1:-all}
HOURS=${2:-}
BRANCH=${TSC_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
USER_NAME=$($KAGGLE config view 2>/dev/null | awk '/username/ {print $NF}')
USER_NAME=${KAGGLE_USERNAME:-${USER_NAME:-hritikshah02}}
SLUG="tsc-train"
BUILD=kaggle/out/push
OUT=kaggle/out/result

if ! git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
  echo "Branch '$BRANCH' is not on GitHub yet - run: git push -u origin $BRANCH" >&2
  exit 1
fi

rm -rf "$BUILD" && mkdir -p "$BUILD" "$OUT"
{
  echo "import os"
  echo "os.environ.update(TSC_BRANCH='$BRANCH', TSC_METHODS='$METHODS', TSC_HOURS='$HOURS')"
  cat kaggle/train_kernel.py
} > "$BUILD/train_kernel.py"
cat > "$BUILD/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/$SLUG",
  "title": "$SLUG",
  "code_file": "train_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
JSON

$KAGGLE kernels push -p "$BUILD"
echo "Waiting for $USER_NAME/$SLUG ..."
while true; do
  STATUS=$($KAGGLE kernels status "$USER_NAME/$SLUG" 2>&1 | tail -1)
  echo "  $(date +%H:%M:%S) $STATUS"
  case "$STATUS" in
    *COMPLETE*|*complete*) break ;;
    *ERROR*|*error*|*CANCEL*|*cancel*) $KAGGLE kernels output "$USER_NAME/$SLUG" -p "$OUT" || true; exit 1 ;;
  esac
  sleep 60
done
$KAGGLE kernels output "$USER_NAME/$SLUG" -p "$OUT"
if [ "${TSC_COPY:-1}" = 1 ]; then
  cp -v "$OUT"/models/* models/
else
  echo "Models saved to $OUT/models (models/ left unchanged)"
fi

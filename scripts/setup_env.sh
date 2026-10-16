#!/bin/bash
LOOPCD_ROOT=${LOOPCD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export LOOPCD_ROOT

export PYTHONPATH="$LOOPCD_ROOT:$LOOPCD_ROOT/third_party/lm-evaluation-harness:$LOOPCD_ROOT/third_party/evalplus:${PYTHONPATH:-}"

echo "[LoopCD] LOOPCD_ROOT=$LOOPCD_ROOT"
echo "[LoopCD] PYTHONPATH:"
echo "  - $LOOPCD_ROOT"
echo "  - $LOOPCD_ROOT/third_party/lm-evaluation-harness  (vendored lm_eval)"
echo "  - $LOOPCD_ROOT/third_party/evalplus                (vendored evalplus)"

#!/bin/bash
# Operator-level A/B, round-level ABBA: base, patch, patch, base.
#
# Why ABBA and not A/B: this machine drifts within a single run. On one sweep
# the *unchanged* arm read 357.6us then 371.5us for the same shape and the same
# package -- 3.9% apart. A plain A-then-B comparison silently books that drift
# as an effect. Averaging the two base passes and the two patch passes cancels
# any drift that is linear in time, per shape.
#
# Usage:
#   GDR_DEV=0 VENDOR=gdrcust_transformer bash bench_op_abba.sh
#
# Requires probe2.py next to this script and the custom vendor package already
# installed (see REPRODUCE.md step 2).
set -u

DEV=${GDR_DEV:-0}
VENDOR=${VENDOR:-gdrcust_transformer}
HERE=$(cd "$(dirname "$0")" && pwd)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
V=$ASCEND_OPP_PATH/vendors
CFG=$V/config.ini

# Qwen3.6-35B-A3B GDN shapes (Dk=Dv=128, global Nk=16 / Nv=32) at each TP degree.
SHAPES="8192_1_2_4 8192_16_2_4 2560_40_2_4 8192_1_4_8 8192_16_4_8 8192_1_8_16 8192_1_16_32"

# Rewrite the whole line rather than stripping "$VENDOR,". With a single custom
# vendor installed there is no trailing comma, so the comma form matches nothing
# and the base arm silently keeps running the patched kernel.
BASE_PRIO=$(sed -n 's/^load_priority=//p' "$CFG" | sed "s/^$VENDOR,//;s/^$VENDOR$//")
use_base()  { echo "load_priority=$BASE_PRIO" > "$CFG"; }
use_patch() { [ -z "$BASE_PRIO" ] && echo "load_priority=$VENDOR" > "$CFG" \
              || echo "load_priority=$VENDOR,$BASE_PRIO" > "$CFG"; }

sweep() {  # $1 = arm label, $2 = 1 to put the vendor's op_api on LD_LIBRARY_PATH
  for s in $SHAPES; do
    args=${s//_/ }
    if [ "$2" = "1" ]; then
      out=$(LD_LIBRARY_PATH=$V/$VENDOR/op_api/lib/:${LD_LIBRARY_PATH:-} \
            GDR_DEV=$DEV python "$HERE/probe2.py" $args 2>/dev/null)
    else
      out=$(GDR_DEV=$DEV python "$HERE/probe2.py" $args 2>/dev/null)
    fi
    echo "$1 $out"
  done
}

echo "### round-level ABBA on npu:$DEV   $(date '+%F %T')"
echo "--- pass1 base  ---"; use_base;  sweep A1base  0
echo "--- pass2 patch ---"; use_patch; sweep B1patch 1
echo "--- pass3 patch ---";            sweep B2patch 1
echo "--- pass4 base  ---"; use_base;  sweep A2base  0
use_base
echo "DONE $(date '+%F %T')"
echo
echo "Report base as mean(A1,A2) and patch as mean(B1,B2), per shape."
echo "Also report the within-arm spread; a delta smaller than it is not a result."

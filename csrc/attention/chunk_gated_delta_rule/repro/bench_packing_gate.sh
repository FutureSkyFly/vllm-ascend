#!/bin/bash
# What the cross-sequence packing gate is worth, and when it engages.
#
# The gate is in chunk_gated_delta_rule.h Process():
#     bool packed = tiling_->b > 1;
#     for each bid: if (bid + 1 < b && length % chunkSize != 0) packed = false;
# so it needs every sequence but the last to be a multiple of 64. Every
# multi-request row in REPRODUCE.md section 1 was measured with
# asl = [T//B]*B, which always satisfies that. This script holds the shape
# fixed and moves only the length vector:
#
#   even     [T//B]*B                  packed=1   the configuration section 1 used
#   lastoff  last length -1            packed=1   control: a partial chunk exists,
#                                                 the gate ignores the last sequence
#   off1     first -1, second +1       packed=0   one token moved, same total
#   jit      every length jittered     packed=0   what real traffic looks like
#
# `off1` versus `lastoff` is the discriminating pair. Not the same work: lastoff
# adds one partial chunk on the last sequence, off1 adds two and therefore one
# extra chunk. What they isolate is the gate. The base arm has no packing at all,
# so its spread across the four modes -- under 2.2% -- prices the work difference.
#
# The B=1 shape is an internal control -- `packed` is false there in every mode,
# so its four numbers per pass estimate the noise floor.
#
# Usage (inside the container, vendor package files already installed):
#   GDR_DEV=0 BASE_RUN=/work/pkg_base.run PATCH_RUN=/work/pkg_patch.run \
#     bash bench_packing_gate.sh
#
# An atomic lock guards against a retrying launcher starting a second instance:
# two instances would install different packages under each other and interleave
# into one log. That invalidated a whole run of this experiment once.
set -u
LOCK=${GDR_LOCK:-/tmp/gdr_packing_gate.lock}
mkdir "$LOCK" 2>/dev/null || { echo "ALREADY RUNNING (lock $LOCK), exiting"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

HERE=$(cd "$(dirname "$0")" && pwd)
DEV=${GDR_DEV:-0}
VENDOR=${VENDOR:-gdrcust_transformer}
BASE_RUN=${BASE_RUN:?set BASE_RUN to the unmodified-source .run package}
PATCH_RUN=${PATCH_RUN:?set PATCH_RUN to the branch .run package}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
V=$ASCEND_OPP_PATH/vendors
CFG=$V/config.ini
LDV="$V/$VENDOR/op_api/lib/"

# The .run installer only writes load_priority when config.ini does not already
# exist. Once the file is there -- for instance left with an empty value by a
# previous built-in arm -- installing copies the files but leaves the vendor
# INACTIVE, and the arm silently measures the CANN built-in instead. Always set
# it explicitly and echo it.
use_pkg() {
  "$1" --quiet >/dev/null 2>&1
  echo "load_priority=$VENDOR" > "$CFG"
}
kmd5() {
  f=$(find "$V/$VENDOR" -name chunk_gated_delta_rule_matmul_basic.h 2>/dev/null | head -1)
  [ -n "$f" ] && md5sum "$f" | cut -c1-8 || echo "absent(base)"
}

SHAPES=${SHAPES:-"8192_16_4_8 8192_16_2_4 2560_40_2_4 8192_1_4_8"}
MODES=${MODES:-"even lastoff off1 jit"}

sweep() {
  for s in $SHAPES; do
    for m in $MODES; do
      out=$(ASL_MODE=$m LD_LIBRARY_PATH=$LDV:${LD_LIBRARY_PATH:-} GDR_DEV=$DEV \
            python3 "$HERE/probe3.py" ${s//_/ } 2>/dev/null | tail -1)
      echo "$1 $out"
    done
  done
}

echo "### packing-gate experiment  npu:$DEV  $(date '+%F %T')"
for pass in P1base P2patch P3patch P4base; do
  case $pass in
    *base)  use_pkg "$BASE_RUN"  ;;
    *patch) use_pkg "$PATCH_RUN" ;;
  esac
  echo "--- $pass   config.ini=[$(cat $CFG)]  kernel.h md5=$(kmd5) ---"
  sweep $pass
done
use_pkg "$BASE_RUN"
echo "PACK_EXP_DONE  $(date '+%F %T')"
echo
echo "Report base as mean(P1,P4) and patch as mean(P2,P3), per shape AND per mode."
echo "The base arm must be flat across modes; if it is not, the difference is"
echo "the shape, not the code path, and the experiment says nothing."

#!/bin/bash
# What this kernel is worth on the batches vLLM actually produces.
#
# Every other operator benchmark here uses a synthetic length vector. The ones
# in real_seq_lengths.txt were captured from an instrumented gdn.py on a live
# server, so the batch count and the ragged tail are the real thing.
#
# Three arms per vector, round-level ABBA over base/patch:
#   base   unmodified kernel, real lengths
#   patch  branch kernel, real lengths                 -> packed=0, what ships today
#   padP   branch kernel, lengths rounded up to 64     -> packed=1, the gate satisfied
#
# padP is a LOWER bound on what fixing the gate would give: it pays for 3.9-7.4%
# extra pad tokens that a real fix would not.
#
# Usage:
#   GDR_DEV=0 BASE_RUN=/work/pkg_base.run PATCH_RUN=/work/pkg_patch.run \
#     bash bench_real_vectors.sh
set -u
LOCK=${GDR_LOCK:-/tmp/gdr_real_vectors.lock}
mkdir "$LOCK" 2>/dev/null || { echo "ALREADY RUNNING (lock $LOCK), exiting"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

HERE=$(cd "$(dirname "$0")" && pwd)
DEV=${GDR_DEV:-0}
VENDOR=${VENDOR:-gdrcust_transformer}
VECFILE=${VECFILE:-$HERE/real_seq_lengths.txt}
BASE_RUN=${BASE_RUN:?set BASE_RUN to the unmodified-source .run package}
PATCH_RUN=${PATCH_RUN:?set PATCH_RUN to the branch .run package}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
V=$ASCEND_OPP_PATH/vendors
CFG=$V/config.ini
LDV="$V/$VENDOR/op_api/lib/"

use_pkg() {
  "$1" --quiet >/dev/null 2>&1
  echo "load_priority=$VENDOR" > "$CFG"    # the installer will not do this for you
}
kmd5() {
  f=$(find "$V/$VENDOR" -name chunk_gated_delta_rule_matmul_basic.h 2>/dev/null | head -1)
  [ -n "$f" ] && md5sum "$f" | cut -c1-8 || echo "absent(base)"
}

run_all() {   # $1 = pass label, $2 = 1 to also run the padded variant
  grep -v '^#' "$VECFILE" | grep '|' | while IFS='|' read -r tag nk nv lens; do
    out=$(LD_LIBRARY_PATH=$LDV:${LD_LIBRARY_PATH:-} GDR_DEV=$DEV \
          python3 "$HERE/probe5.py" "$nk" "$nv" "$lens" nopad "$tag" 2>/dev/null | tail -1)
    echo "$1 real  $out"
    if [ "$2" = "1" ]; then
      out=$(LD_LIBRARY_PATH=$LDV:${LD_LIBRARY_PATH:-} GDR_DEV=$DEV \
            python3 "$HERE/probe5.py" "$nk" "$nv" "$lens" pad "$tag" 2>/dev/null | tail -1)
      echo "$1 padP  $out"
    fi
  done
}

echo "### real-batch packing value  npu:$DEV  $(date '+%F %T')"
for pass in P1base P2patch P3patch P4base; do
  case $pass in
    *base)  use_pkg "$BASE_RUN";  PAD=0 ;;
    *patch) use_pkg "$PATCH_RUN"; PAD=1 ;;
  esac
  echo "--- $pass  cfg=[$(cat $CFG)]  md5=$(kmd5) ---"
  run_all $pass $PAD
done
use_pkg "$BASE_RUN"
echo "REALVEC_DONE  $(date '+%F %T')"

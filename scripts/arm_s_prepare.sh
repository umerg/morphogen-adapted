#!/bin/bash
# Arm S prep, end to end. Run this on a login node; it needs no GPU.
#
#   bash scripts/arm_s_prepare.sh
#
# What it does:
#   1. finds the baked corpus and reports the cell-class histogram
#   2. builds a 690-neuron class-0 (23P) subset synset -- copies, no re-baking
#   3. measures the GT cloud baseline that Arm S will be judged against
#   4. submits the throughput PROBE (40 epochs, writes no checkpoints, then exits)
#
# Class 0 = 23P is chosen because it is by far the most populous (7,796 train
# neurons), so 690 is a random sample of a large pool rather than nearly all of a
# small one. Names: utils/data_loading.py:12 CELL_CLASS_NAMES.
#
# Set SUBMIT=0 to do the prep and print the sbatch line instead of submitting.

set -o errexit
set -o pipefail

ETH_USERNAME=${ETH_USERNAME:-guptau}
SHARE=/itet-stor/${ETH_USERNAME}/net_scratch
CELL_CLASS=${CELL_CLASS:-0}          # 0 = 23P
N=${N:-690}                          # paper: ~690-760 train neurons per model
SEED=${SEED:-0}
SUBMIT=${SUBMIT:-1}

OUT=${OUT:-${SHARE}/morphogen_npy_c${CELL_CLASS}_${N}}
EXP=${EXP:-armU_S${N}_c${CELL_CLASS}}

# The corpus may be on the share or only on node-local scratch. A login node cannot
# see a compute node's /scratch, so say which one is missing rather than failing
# later inside the dataloader.
BAKE=${BAKE:-}
if [[ -z ${BAKE} ]]; then
  for cand in ${SHARE}/morphogen_npy /scratch/${ETH_USERNAME}/morphogen_npy; do
    if [[ -d ${cand}/neurons/train ]]; then BAKE=${cand}; break; fi
  done
fi
if [[ -z ${BAKE} ]]; then
  echo "FATAL: no baked corpus found." >&2
  echo "  looked in ${SHARE}/morphogen_npy and /scratch/${ETH_USERNAME}/morphogen_npy" >&2
  echo "  if it is only on a compute node's /scratch, run this inside an interactive" >&2
  echo "  job on that node, or pass BAKE=/path/to/morphogen_npy" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
echo "=== Arm S prep ==="
echo "  bake        ${BAKE}"
echo "  subset out  ${OUT}"
echo "  class       ${CELL_CLASS} (23P)   n=${N}   seed=${SEED}"
echo "  experiment  ${EXP}"
echo

# --- 1 + 2: histogram, then build the subset --------------------------------
python -u tools/make_subset_synset.py \
  --npy-root "${BAKE}" \
  --out-root "${OUT}" \
  --cell-class "${CELL_CLASS}" \
  --n "${N}" --seed "${SEED}"

TRAIN_N=$(ls "${OUT}"/neurons/train/*.npy | wc -l)
STEPS=$(( TRAIN_N / 112 ))
if [[ ${STEPS} -lt 1 ]]; then
  # drop_last=True, so fewer than one full batch per epoch means the trainer would
  # spin through every epoch doing nothing at all. Stop here rather than submit a
  # job with an empty NITER.
  echo "FATAL: ${TRAIN_N} train neurons is under one batch at bs 112 (drop_last)," >&2
  echo "  so there would be 0 steps per epoch. Raise N or lower BS." >&2
  exit 1
fi
if [[ ${STEPS} -ne 6 ]]; then
  echo "NOTE: ${STEPS} steps/epoch at bs 112, not the expected 6 -- NITER is derived" >&2
  echo "  from it below, so Arm P's 60,900-step budget is still matched." >&2
fi
NITER=$(( 60900 / STEPS ))

echo
echo "=== budget ==="
echo "  ${TRAIN_N} train neurons / bs 112 = ${STEPS} steps/epoch"
echo "  NITER=${NITER} epochs x ${STEPS} = $(( NITER * STEPS )) steps  (Arm P: 60,900)"
echo "  per-neuron exposure ${NITER} passes  (Arm P: 300, paper: 40,000)"

# --- 3: the GT baseline Arm S is judged against ------------------------------
# Measured on the CLUSTER bake, which comes from the raw degree-2 corpus. A value
# taken from a cleaned-corpus bake is inflated -- its interpolated straight segments
# are artificially filamentary -- and would overstate the gap.
echo
echo "=== GT cloud baseline (class ${CELL_CLASS} val, 200 neurons) ==="
python -u tools/cloud_quality.py \
  --ref-npy-root "${OUT}/neurons" --ref-split val --limit 200 \
  --output-json "${OUT}/gt_baseline.json"

echo
echo "=== next ==="
PROBE_CMD="DATA_SRC=${OUT} EXP=${EXP} NITER=${NITER} SAVEITER=500 BS=112 \
PROBE=1 PROBE_EPOCHS=40 sbatch scripts/train_armU.sbatch"
TRAIN_CMD="DATA_SRC=${OUT} EXP=${EXP} NITER=${NITER} SAVEITER=500 BS=112 \
sbatch scripts/train_armU.sbatch"

if [[ ${SUBMIT} -ne 0 ]]; then
  echo "submitting the probe (40 epochs, no checkpoints, exits on its own):"
  echo "  ${PROBE_CMD}"
  DATA_SRC="${OUT}" EXP="${EXP}" NITER="${NITER}" SAVEITER=500 BS=112 \
    PROBE=1 PROBE_EPOCHS=40 sbatch scripts/train_armU.sbatch
  echo
  echo "when the probe reports 'steps/epoch: ${STEPS}' and a sane projection, train with:"
else
  echo "probe (run this first):"
  echo "  ${PROBE_CMD}"
  echo
  echo "then train with:"
fi
echo "  ${TRAIN_CMD}"

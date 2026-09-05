#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mg_sweep_cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#
# Phase 2a/3, CPU half: reconstruct -> finalize -> score every sampled candidate, then
# emit the selection curve.
#
#   sbatch scripts/sweep_score.sbatch
#
# NO --gres: this stage never touches the GPU and never loads a checkpoint, which is
# the whole point of the split -- CPU nodes are far easier to get than GPU ones, and
# reconstruction (2.9 s/neuron) is the long half.
#
# Resumable twice over: per candidate (a dist.json already on disk is skipped) and per
# neuron inside reconstruction, so an interrupted job restarts where it stopped.
#
# Selection is on headline_excess_mmd_morpho over VAL. Test is never used to choose a
# checkpoint; it is scored once, at the end, on the winner.

ETH_USERNAME=guptau
PROJECT_NAME=morphogen-adapted
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=MOPRHO2
SHARE=/itet-stor/${ETH_USERNAME}/net_scratch

EXP=${EXP:-armU}
RUNS_DIR=${RUNS_DIR:-${SHARE}/mg_runs/${EXP}}
OUT_DIR=${OUT_DIR:-${SHARE}/mg_sweep/${EXP}}
DATAROOT=${DATAROOT:-/scratch/${ETH_USERNAME}/morphogen_npy}
[[ -d ${DATAROOT}/neurons/train ]] || DATAROOT=${SHARE}/morphogen_npy
CORPUS=${CORPUS:-${SHARE}/neurons_conditional_full}
EVERY=${EVERY:-3}
EPOCHS=${EPOCHS:-}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

# MorphoGen+ reconstruction settings: train-calibrated, confirmed on held-out val at
# 4.14x. Drop GAMMA_SEED/TAU overrides to their shipped values for the faithful arm.
DETECT_RADIUS=${DETECT_RADIUS:-0.30}
ROOT_CAP=${ROOT_CAP:-23}
GAMMA_SEED=${GAMMA_SEED:-0.40}
TAU=${TAU:-0.30}

set -o errexit
set -o pipefail
trap 'echo "=== caught TERM; scored candidates are on SHARE, resubmit to resume ==="' TERM

echo "Running on node: $(hostname)"; echo "Starting: $(date)"

[[ -f ${SHARE}/conda/bin/conda ]] && eval "$(${SHARE}/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
cd ${DIRECTORY}

# clean_swc_tree and compute_distribution_metrics are dendrite_gen's, imported lazily.
# Check here rather than letting it surface hours in: the default only ever resolves on
# a dev laptop, so an unset DENDRITE_GEN is the normal failure on a cluster.
export DENDRITE_GEN=${DENDRITE_GEN:-${SHARE}/dendrite_gen}
if [[ ! -f ${DENDRITE_GEN}/validation/dist_metrics.py ]]; then
  echo "FATAL: no dendrite_gen at ${DENDRITE_GEN}" >&2
  echo "       export DENDRITE_GEN=/path/to/dendrite_gen and resubmit." >&2
  exit 1
fi
for d in "${OUT_DIR}" "${CORPUS}/val" "${CORPUS}/train" "${DATAROOT}/neurons/train"; do
  [[ -d ${d} ]] || { echo "FATAL: missing ${d}" >&2; exit 1; }
done

ARGS=(--phase score --runs-dir "${RUNS_DIR}" --out-dir "${OUT_DIR}"
      --gt-root "${CORPUS}/val" --train-dir "${CORPUS}/train"
      --train-npy-root "${DATAROOT}/neurons" --workers ${WORKERS}
      --scale-mode marginal
      --detect-radius ${DETECT_RADIUS} --root-cap ${ROOT_CAP}
      --gamma-seed ${GAMMA_SEED} --length-threshold ${TAU})
[[ -n ${EPOCHS} ]] && ARGS+=(--epochs "${EPOCHS}") || ARGS+=(--every ${EVERY})

python -u tools/selection_sweep.py "${ARGS[@]}"

cat <<NEXT

=== read the curve ===
  ${OUT_DIR}/curve.json   ${OUT_DIR}/curve.png   ${OUT_DIR}/best.json

If the curve is still falling at the last epoch, extend training -- it is a
continuation, not a retrain:
  NITER=600 sbatch scripts/train_armU.sbatch
If it is flat, the run has converged: report the curve as the answer to
"undertrained" and take the last checkpoint.
NEXT
echo "Finished: $(date)"

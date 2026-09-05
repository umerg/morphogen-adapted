#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mg_sweep_gpu
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --exclude=tikgpu10
#
# Phase 2a/3, GPU half: sample N clouds from each candidate checkpoint.
#
#   sbatch scripts/sweep_sample.sbatch          # then sweep_score.sbatch on CPU
#   sbatch scripts/sweep_sample.sbatch          # again -> resumes, skips what exists
#
# Sampling is 1000 sequential denoising steps per batch and is the only part of the
# sweep that wants a GPU; reconstruction and scoring are pure CPU and run as a
# separate job, so a GPU allocation is never held while a CPU-bound stage runs.
#
# Resumable at candidate granularity: a checkpoint whose clouds.json already exists is
# skipped, so an overrun costs at most one candidate.
#
# N MUST BE CONSTANT ACROSS CANDIDATES -- that, not its size, is what makes the
# comparison valid. Change it and the whole sweep must be re-run, not topped up.

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
EVERY=${EVERY:-3}
EPOCHS=${EPOCHS:-}
N=${N:-512}
BS=${BS:-50}

set -o errexit
set -o pipefail
trap 'echo "=== caught TERM; sampled candidates are on SHARE, resubmit to resume ==="' TERM

echo "Running on node: $(hostname)"; echo "Starting: $(date)"

[[ -f ${SHARE}/conda/bin/conda ]] && eval "$(${SHARE}/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
cd ${DIRECTORY}

[[ -d ${RUNS_DIR} ]] || { echo "FATAL: no checkpoint dir ${RUNS_DIR}" >&2; exit 1; }
[[ -d ${DATAROOT}/neurons/train ]] || { echo "FATAL: no bake at ${DATAROOT}" >&2; exit 1; }

python -u -c "import torch; print('torch', torch.__version__, '|',
      torch.cuda.get_device_name(0),
      '| %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))"

# One visible device: nn.DataParallel takes its single-device fast path and skips the
# per-step replicate/scatter/gather. With several visible it would also replicate onto
# cards other jobs hold.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(--phase sample --runs-dir "${RUNS_DIR}" --out-dir "${OUT_DIR}"
      --dataroot "${DATAROOT}" --n ${N} --bs ${BS} --sample-workers 1)
[[ -n ${EPOCHS} ]] && ARGS+=(--epochs "${EPOCHS}") || ARGS+=(--every ${EVERY})

python -u tools/selection_sweep.py "${ARGS[@]}"

echo "=== sampled ==="
ls -d "${OUT_DIR}"/ep* 2>/dev/null | wc -l
cat <<NEXT

=== next: the CPU half ===
  sbatch scripts/sweep_score.sbatch     # reconstruct -> finalize -> score, no GPU
NEXT
echo "Finished: $(date)"

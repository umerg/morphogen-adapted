#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=tikgpu10
#
# Stage-4 GATE: exercises every interface on a small subset. The point is
# interfaces, not numbers -- do not request full-run hours until this is green.
#
#   sbatch scripts/dryrun.sbatch
#
# NOTE: torch must be a cu121 build. torch>=2.6 dropped cu121 wheels, so on a
# 12.0-12.3 driver use torch==2.5.1 + torchvision==0.20.1. A newer build fails at
# .cuda() with "NVIDIA driver ... too old", which means the BUILD is too new.

ETH_USERNAME=guptau
PROJECT_NAME=morphogen-adapted
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=MOPRHO2
SCRATCH=/itet-stor/${ETH_USERNAME}/net_scratch

set -o errexit

TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then echo 'Failed to create temp directory' >&2; exit 1; fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

echo "Running on node: $(hostname)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
  eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
cd ${DIRECTORY}

python -u -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '|', torch.cuda.get_device_name(0))"

RUNS=${SCRATCH}/mg_runs
GEN=${SCRATCH}/mg_gen/dryrun

# --- train -------------------------------------------------------------------
# --bs MUST be <= the number of training neurons: the loader uses drop_last=True,
# so a batch larger than the dataset yields ZERO batches and the loop silently
# completes having done nothing.
python -u DDPM_train.py \
  --dataroot "${SCRATCH}/morphogen_npy" --category neurons \
  --model_dir "${RUNS}" --experiment_name dryrun \
  --bs 64 --niter 20 --saveIter 5 --use_ema \
  --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1

echo "=== checkpoints ==="
ls -la "${RUNS}/dryrun"/*.pth
python -u - <<PY
import glob, torch
for f in sorted(glob.glob('${RUNS}/dryrun/*.pth')):
    k = torch.load(f, map_location='cpu').keys()
    print(f.split('/')[-1], '->', sorted(k), '| ema present:', 'ema' in k)
PY

# --- generate ----------------------------------------------------------------
# MorphoGen+ reconstruction settings (calibrated on train, confirmed on held-out
# val). Drop --gamma_seed/--length_threshold for the faithful arm.
python -u morphology_gen.py \
  --dataroot "${SCRATCH}/morphogen_npy" --category neurons \
  --model "${RUNS}/dryrun/epoch_19.pth" \
  --generate_dir "${GEN}" \
  --bs 16 --num_classes 1 \
  --detect_radius 0.30 --root_cap 23 --gamma_seed 0.40 --length_threshold 0.30

echo "=== generated output ==="
python -u - <<PY
import json, glob, os
d = '${GEN}'
swc = glob.glob(os.path.join(d, '*.swc'))
man = os.path.join(d, 'manifest.json')
print('swc files:', len(swc), '| manifest:', os.path.isfile(man))
if os.path.isfile(man):
    m = json.load(open(man))
    mids = [r['mid'] for r in m]
    print('manifest rows:', len(m), '| unique mids:', len(set(mids)),
          '| bijection:', len(m) == len(set(mids)) == len(swc))
    print('sample:', {k: m[0][k] for k in ('file', 'mid', 'cate_idx')})
PY

echo "Finished at: $(date)"
exit 0

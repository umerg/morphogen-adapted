#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu10
#SBATCH --nodelist=tikgpu07 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

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

# The corpus lives on node-local scratch (fast, and where the bake left it), so
# that is tried first and the share is the fallback. Checkpoints and generated
# output always go to SHARE: they are small, and node-local scratch does not
# survive the node. Override any of these from the environment.
SHARE=/itet-stor/${ETH_USERNAME}/net_scratch
DATA_SRC=${DATA_SRC:-/scratch/${ETH_USERNAME}/morphogen_npy}
if [[ ! -d ${DATA_SRC}/neurons/train ]]; then
  DATA_SRC=${SHARE}/morphogen_npy
fi
RUNS=${RUNS:-${SHARE}/mg_runs}
GEN=${GEN:-${SHARE}/mg_gen/dryrun}

# NOTE: if the corpus is on node-local scratch, this job must land on the node
# that holds it -- set #SBATCH --nodelist to that node (arton10 as of the last
# bake), or point DATA_SRC at the share and let the script stage it.

# Stage-4 is an interface gate, so it runs on a subset: the full corpus loads
# 4.1 GB into RAM and 20 epochs over it is a real training run, not a smoke test.
# Set SUBSET=0 to use the whole corpus.
SUBSET=${SUBSET:-200}

set -o errexit

if [[ -d /scratch && -w /scratch ]]; then
  LOCAL=/scratch/${ETH_USERNAME}/dryrun.${SLURM_JOB_ID:-$$}
else
  LOCAL=$(mktemp -d)/dryrun
fi
mkdir -p "${LOCAL}"
export TMPDIR="${LOCAL}"
trap "exit 1" HUP INT TERM
trap 'rm -rf "${LOCAL}"' EXIT

echo "Running on node: $(hostname)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
  eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
cd ${DIRECTORY}

python -u -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '|', torch.cuda.get_device_name(0))"

# --- stage the data ----------------------------------------------------------
if [[ ! -d ${DATA_SRC}/neurons/train ]]; then
  echo "FATAL: no baked corpus at ${DATA_SRC}/neurons/train" >&2
  echo "       Checked node-local scratch and ${SHARE}/morphogen_npy." >&2
  echo "       If it is on another node's /scratch, pin this job there with --nodelist." >&2
  exit 1
fi
echo "corpus: ${DATA_SRC}"
DATA=${LOCAL}/data
mkdir -p "${DATA}/neurons/train" "${DATA}/neurons/val"
if [[ ${SUBSET} -le 0 && ${DATA_SRC} == /scratch/* ]]; then
  # already on this node's disk; use it in place
  DATA=${DATA_SRC}
  echo "=== corpus already node-local, using in place ==="
elif [[ ${SUBSET} -gt 0 ]]; then
  echo "=== staging a ${SUBSET}-neuron subset to ${DATA} ==="
  for sp in train val; do
    n=${SUBSET}; [[ ${sp} == val ]] && n=$(( SUBSET / 5 + 1 ))
    for f in $(ls "${DATA_SRC}/neurons/${sp}"/*.npy | head -${n}); do
      cp "${f}" "${f%.npy}.meta.json" "${DATA}/neurons/${sp}/" 2>/dev/null || cp "${f}" "${DATA}/neurons/${sp}/"
    done
  done
else
  echo "=== staging the full corpus to ${DATA} ==="
  rsync -a "${DATA_SRC}/" "${DATA}/"
fi
echo "using: train $(ls ${DATA}/neurons/train/*.npy | wc -l), val $(ls ${DATA}/neurons/val/*.npy | wc -l)"

# --- train -------------------------------------------------------------------
# --workers defaults to 16, which oversubscribes a --cpus-per-task=4 allocation
# and makes torch warn; track the allocation instead.
#
# --bs MUST be <= the number of training neurons: the loader uses drop_last=True,
# so a batch larger than the dataset yields ZERO batches and the loop silently
# completes having done nothing.
python -u DDPM_train.py \
  --dataroot "${DATA}" --category neurons \
  --model_dir "${RUNS}" --experiment_name dryrun \
  --bs 64 --niter 20 --saveIter 5 --use_ema \
  --workers ${SLURM_CPUS_PER_TASK:-4} \
  --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1

echo "=== checkpoints ==="
ls -la "${RUNS}/dryrun"/*.pth
python -u - <<PY
import glob, torch
for f in sorted(glob.glob('${RUNS}/dryrun/*.pth')):
    k = torch.load(f, map_location='cpu').keys()
    print(f.split('/')[-1], '->', sorted(k), '| ema present:', 'ema' in k)
    if 'ema' not in k:
        raise SystemExit('GATE FAILED: checkpoint has no ema; --use_ema did not take')
PY

# --- generate ----------------------------------------------------------------
# MorphoGen+ reconstruction settings (calibrated on train, confirmed on held-out
# val). Drop --gamma_seed/--length_threshold for the faithful arm.
python -u morphology_gen.py \
  --dataroot "${DATA}" --category neurons \
  --model "${RUNS}/dryrun/epoch_19.pth" \
  --generate_dir "${GEN}" \
  --bs 16 --num_classes 1 --workers ${SLURM_CPUS_PER_TASK:-4} \
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
    if not (len(m) == len(set(mids)) == len(swc)):
        raise SystemExit('GATE FAILED: manifest is not a bijection onto the generated SWCs')
elif not swc:
    raise SystemExit('GATE FAILED: no SWC files were generated')
PY

echo "Finished at: $(date)"
exit 0

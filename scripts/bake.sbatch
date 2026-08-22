#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --exclude=tikgpu10
#
# Bake SWC -> (15000,3) point clouds.  CPU ONLY -- no --gres=gpu here on purpose.
#
# Cost is ~6 s/neuron/worker, dominated by the farthest-point-sample down to 15000
# from the ~39k-point upsampled cloud. For all 26,469 neurons:
#     16 cpus -> ~2.8 h      32 cpus -> ~1.4 h      64 cpus -> ~42 min
# Scales close to linearly, so ask for as many cpus as the queue will give.
#
#   mkdir -p /itet-stor/guptau/net_scratch/cluster/jobs
#   sbatch scripts/bake.sbatch
#   tail -f /itet-stor/guptau/net_scratch/cluster/jobs/<jobid>.out

ETH_USERNAME=guptau
PROJECT_NAME=morphogen-adapted
# NOTE: adjust if your checkout lives elsewhere -- the running copy was last seen at
# /usr/itetnas04/data-scratch-01/guptau/data/morphogen-adapted
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
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
  eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated: $(python --version 2>&1)"

cd ${DIRECTORY}

# -u keeps stdout unbuffered so `tail -f` shows live progress; without it the job
# looks hung for tens of minutes while working normally.
python -u tools/swc_to_morphogen_npy.py \
  --raw-root   "${SCRATCH}/neurons_raw" \
  --label-root "${SCRATCH}/neurons_conditional_full" \
  --out-root   "${SCRATCH}/morphogen_npy" \
  --mode uncond \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

echo "=== verifying ==="
python -u - <<'PY'
import glob, json, numpy as np
root = '/itet-stor/guptau/net_scratch/morphogen_npy/neurons'
tot = 0
for split, want in (('train', 22773), ('val', 2529), ('test', 1167)):
    f = sorted(glob.glob(f'{root}/{split}/*.npy')); tot += len(f)
    if not f:
        print(f'{split:6} MISSING'); continue
    a = np.load(f[0]); m = json.load(open(f[0].replace('.npy', '.meta.json')))
    ok = a.shape == (15000, 3) and bool(np.isfinite(a).all()) and 'scale_m' in m
    flag = 'ok' if (ok and len(f) == want) else 'CHECK'
    print(f'{split:6} {len(f):6d}/{want:<6} {a.shape} {a.dtype}  sidecar={flag}')
print('total', tot, '/ 26469')
PY

echo "Finished at: $(date)"
exit 0

#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --exclude=tikgpu10
#
# Bake SWC -> (15000,3) point clouds.  CPU ONLY -- no --gres=gpu here on purpose.
#
# Runs entirely on node-local disk: the corpus is staged in, baked locally, and
# the result copied back. The first attempt at this ran straight off /itet-stor
# and took ~11 h (0.67 neurons/s aggregate across 32 workers, i.e. ~48 s of
# wall-clock per neuron per worker against ~6 s measured on a local SSD) -- the
# job was bound by NFS metadata traffic, not by CPU. It writes two small files
# per neuron, 53k round trips.
#
# The copy-back runs from the EXIT trap, so a job killed by the time limit still
# leaves its finished neurons on the share; resubmitting with --skip-existing
# then picks up where it stopped instead of restarting from zero.
#
#   mkdir -p /itet-stor/guptau/net_scratch/cluster/jobs
#   sbatch scripts/bake.sbatch
#   tail -f /itet-stor/guptau/net_scratch/cluster/jobs/<jobid>.out

ETH_USERNAME=guptau
PROJECT_NAME=morphogen-adapted
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=MOPRHO2

# These are the SHARE paths, not node-local ones. The script stages the corpus
# onto node-local disk itself (see LOCAL below) and copies the result back here.
# Pointing these at /scratch/$USER makes it stage local -> local and "copy back"
# to local, which is how the first 10 h bake ended up stranded on arton10.
SHARE=/itet-stor/${ETH_USERNAME}/net_scratch
RAW_SRC=${RAW_SRC:-${SHARE}/neurons_raw}
LABEL_SRC=${LABEL_SRC:-${SHARE}/neurons_conditional_full}
OUT_DST=${OUT_DST:-${SHARE}/morphogen_npy}

set -o errexit
set -o pipefail

echo "Running on node: $(hostname)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}"

# ---- pick node-local scratch -------------------------------------------------
# Prefer a real local disk; fall back to mktemp only if /scratch is unusable.
if [[ -d /scratch && -w /scratch ]]; then
  LOCAL=/scratch/${ETH_USERNAME}/bake.${SLURM_JOB_ID:-$$}
else
  LOCAL=$(mktemp -d)/bake
  echo "WARNING: /scratch not writable, falling back to ${LOCAL}" >&2
fi
mkdir -p "${LOCAL}"
export TMPDIR="${LOCAL}"
echo "node-local staging dir: ${LOCAL}"
df -h "${LOCAL}" | tail -1

# ---- copy back before cleaning up -------------------------------------------
# Deliberately NOT a bare `rm -rf`: an earlier version trapped EXIT on a plain
# delete, which is how a completed bake can report success and leave nothing
# behind. Nothing local is removed unless the copy back has succeeded.
finalize() {
  local rc=$?
  set +o errexit
  if [[ -d ${LOCAL}/out ]]; then
    echo "=== copying results back to ${OUT_DST} (exit code so far: ${rc}) ==="
    mkdir -p "${OUT_DST}"
    rsync -a "${LOCAL}/out/" "${OUT_DST}/"
    local sync_rc=$?
    if [[ ${sync_rc} -eq 0 ]]; then
      echo "copy back ok; removing ${LOCAL}"
      rm -rf "${LOCAL}"
    else
      echo "COPY BACK FAILED (rsync=${sync_rc}) -- LEAVING ${LOCAL} ON $(hostname)" >&2
      echo "recover with: rsync -a ${LOCAL}/out/ ${OUT_DST}/" >&2
      exit 1
    fi
  else
    rm -rf "${LOCAL}"
  fi
  exit ${rc}
}
trap finalize EXIT
trap 'exit 1' HUP INT TERM

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
  eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated: $(python --version 2>&1)"

cd ${DIRECTORY}

# ---- stage the corpus in -----------------------------------------------------
case ${OUT_DST} in
  /scratch/*) echo "WARNING: OUT_DST=${OUT_DST} is node-local and will not survive the node." >&2
              echo "WARNING: copy it to ${SHARE} before the allocation ends." >&2 ;;
esac

if [[ ! -d ${RAW_SRC} ]]; then
  echo "FATAL: raw corpus not found at ${RAW_SRC}" >&2; exit 1
fi
if [[ ! -d ${LABEL_SRC} ]]; then
  # Not fatal for the unconditional arm, but every sidecar would get
  # cell_class=null, and --mode class would then skip all 26,469 neurons.
  echo "WARNING: label corpus not found at ${LABEL_SRC}" >&2
  echo "WARNING: sidecars will have cell_class=null; the class-conditional arm needs this" >&2
fi

echo "=== staging corpus to node-local disk ==="
t0=$(date +%s)
rsync -a "${RAW_SRC}/"   "${LOCAL}/in_raw/"
rsync -a "${LABEL_SRC}/" "${LOCAL}/in_label/"
echo "staged in $(( $(date +%s) - t0 ))s: $(du -sh ${LOCAL}/in_raw | cut -f1) raw"

# Resume any neurons a previous attempt already finished.
mkdir -p "${LOCAL}/out"
if [[ -d ${OUT_DST} ]]; then
  echo "=== pulling previously-baked neurons back for resume ==="
  rsync -a "${OUT_DST}/" "${LOCAL}/out/"
  echo "resuming with $(find ${LOCAL}/out -name '*.npy' | wc -l) already baked"
fi

# -u keeps stdout unbuffered so `tail -f` shows live progress; without it the job
# looks hung for tens of minutes while working normally.
python -u tools/swc_to_morphogen_npy.py \
  --raw-root   "${LOCAL}/in_raw" \
  --label-root "${LOCAL}/in_label" \
  --out-root   "${LOCAL}/out" \
  --mode uncond \
  --skip-existing \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

echo "=== verifying (node-local, before copy back) ==="
python -u - "${LOCAL}/out" <<'PY'
import glob, json, sys
import numpy as np
root = f'{sys.argv[1]}/neurons'
bad, tot = [], 0
for split, want in (('train', 22773), ('val', 2529), ('test', 1167)):
    f = sorted(glob.glob(f'{root}/{split}/*.npy')); tot += len(f)
    if not f:
        print(f'{split:6} MISSING'); bad.append(split); continue
    a = np.load(f[0]); m = json.load(open(f[0].replace('.npy', '.meta.json')))
    ok = a.shape == (15000, 3) and bool(np.isfinite(a).all()) and 'scale_m' in m
    if not (ok and len(f) == want):
        bad.append(split)
    print(f'{split:6} {len(f):6d}/{want:<6} {a.shape} {a.dtype}  sidecar={"ok" if ok else "CHECK"}')
print('total', tot, '/ 26469')
if bad:
    # Exit non-zero so a short bake cannot report success. The EXIT trap still
    # copies whatever was produced back to the share first.
    sys.exit(f'VERIFICATION FAILED: {bad}')
PY

echo "Finished at: $(date)"
exit 0

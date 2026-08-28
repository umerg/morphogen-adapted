#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mg_armU
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --exclude=tikgpu10
#CommentSBATCH --nodelist=tikgpu05
#CommentSBATCH --account=tik-internal
#
# PICKING A GPU ON A MIXED NODE -- --constraint does NOT do this.
#   --constraint matches node FEATURES, which are properties of the whole node.
#   On a node holding both 24 GiB and 32 GiB cards every GPU has the same
#   features, so a constraint selects the node you already have and cannot
#   choose the card. --nodelist has the same problem: tikgpu05 is mixed, so
#   pinning to it still hands out a 24 GiB card sometimes -- which is exactly
#   how the 2026-08-28 run failed twice.
#
#   The mechanism that does work is the GRES TYPE:
#       sbatch --gres=gpu:<type>:1 scripts/train_armU.sbatch
#   overriding the #SBATCH --gres line below without editing this file.
#
#   Find <type> on the cluster -- do not guess, a wrong string pends forever:
#       scontrol show node tikgpu05 | grep -E 'Gres|Features'
#       sinfo -N -o '%N | gres:%G | features:%f'
#   Use the exact lowercase token from `Gres=gpu:<type>:N`, e.g. `v100`, not a
#   marketing name like 'Tesla V100'.
#
#   CAVEAT: V100 ships in 16 GiB and 32 GiB variants. If the cluster exposes
#   both under one type string, the type alone will not guarantee 32 GiB. The
#   preflight below is the backstop either way -- it reads the device that was
#   actually allocated and refuses to start before the dataset load.

#
# Stage 6 -- Arm U (unconditional) training.
#
#   PROBE=1 sbatch scripts/train_armU.sbatch     # measure throughput, then exit
#   sbatch scripts/train_armU.sbatch             # the real run
#   sbatch scripts/train_armU.sbatch             # again -> auto-resumes
#
# RUN THE PROBE FIRST. The training budget has never been measured; the only datum
# is the Stage-4 dry run (~1.6 s/step at bs 64 on a 200-neuron subset with an
# oversubscribed dataloader), which is too contaminated to extrapolate from. The
# probe runs a few epochs on the real corpus and prints the projected wall-clock
# for the full run, so the --time and the arm choice are decided on a measurement.
#
# BS = 128. GPU memory, MEASURED 2026-08-28. The earlier table here was the
# plan's estimate and it was WRONG -- bs 128 does not fit in 24 GB:
#   activations ~0.179 GiB/sample, on top of ~0.49 GiB weights+grads+AdamW.
#   bs  96 -> ~17.6 GiB   fits a 24 GiB card
#   bs 112 -> ~20.5 GiB   fits, tight
#   bs 128 -> ~23.4 GiB   OOMs a 23.64 GiB card; needs >=32 GiB
#   bs 256 -> ~46.2 GiB   would need an a100_80gb
#
# bs 128 therefore needs a 32 GiB card, which on a mixed node must be requested
# by GRES type (see the block above), not by --constraint or --nodelist.
# If a 32 GiB card cannot be reserved reliably, BS=112 (~20.5 GiB) fits any
# 24 GiB card with ~3 GiB spare: 203 steps/epoch, 60,900 steps at E=300. That is
# still epoch parity -- report the step count and move on.
#
# What bs 128 does and does not claim, for the write-up:
#   KEPT     epoch parity, E=300 -- the house rule from
#            docs/BUDGET_PARITY_SEMLAFLOW.md is "match epochs, REPORT steps".
#   LOST     the secondary batch-size match with dendrite_gen (128 vs 256) and
#            with it the "within 35% on steps" claim: 300 x 177 = 53,100 steps
#            against dendrite_gen's 20,000, i.e. 2.7x. Report the number; do not
#            describe this arm as step-matched.
#   NOTE     more, noisier updates at the same epoch budget is not a handicap for
#            a diffusion model -- if anything it favours the baseline, which is
#            the safe direction. lr stays at the published 1e-4; linear scaling
#            would argue for 5e-5, but retuning the baseline's optimiser is an
#            extra deviation we have no evidence we need.
#
# NOTE: torch must be a cu121 build. torch>=2.6 dropped cu121 wheels, so on a
# 12.0-12.3 driver use torch==2.5.1 + torchvision==0.20.1. A newer build fails at
# .cuda() with "NVIDIA driver ... too old", which means the BUILD is too new.

ETH_USERNAME=guptau
PROJECT_NAME=morphogen-adapted
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=MOPRHO2

SHARE=/itet-stor/${ETH_USERNAME}/net_scratch
DATA_SRC=${DATA_SRC:-/scratch/${ETH_USERNAME}/morphogen_npy}
if [[ ! -d ${DATA_SRC}/neurons/train ]]; then
  DATA_SRC=${SHARE}/morphogen_npy
fi
RUNS=${RUNS:-${SHARE}/mg_runs}
EXP=${EXP:-armU_P_bs128}   # bs in the name: a later bs-256 arm must not
                           # land in the same checkpoint directory

# Arm P (parity): E=300 at bs 128 = ~53,100 steps on 22,773 neurons (drop_last).
# Arm L (best-effort): set NITER higher and stop when the selection curve flattens.
BS=${BS:-128}
NITER=${NITER:-300}
SAVEITER=${SAVEITER:-10}          # selection grid must be at least as fine as this
PROBE=${PROBE:-0}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}

set -o errexit
set -o pipefail

if [[ -d /scratch && -w /scratch ]]; then
  LOCAL=/scratch/${ETH_USERNAME}/armU.${SLURM_JOB_ID:-$$}
else
  LOCAL=$(mktemp -d)/armU
fi
mkdir -p "${LOCAL}"
export TMPDIR="${LOCAL}"

# LOCAL is always disposable: it holds either a staging COPY of the corpus or just
# TMPDIR, never the corpus itself (DATA_SRC is a different path by construction).
# Checkpoints live on SHARE, so a kill at any point loses at most SAVEITER epochs
# and the next submission resumes. Guarded against an unset/rootish LOCAL rather
# than against DATA_SRC, which rm can never reach from here.
cleanup() {
  case "${LOCAL:-}" in
    /scratch/*|/tmp/*|/var/folders/*) rm -rf "${LOCAL}" ;;
    *) echo "not removing unexpected LOCAL=${LOCAL:-<unset>}" >&2 ;;
  esac
}
trap cleanup EXIT
trap 'echo "=== caught TERM (time limit approaching); checkpoints are on SHARE, resubmit to resume ==="' TERM

echo "Running on node: $(hostname)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f ${SHARE}/conda/bin/conda ]] && eval "$(${SHARE}/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
cd ${DIRECTORY}

python -u -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '|', torch.cuda.get_device_name(0),
      '| %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))"

# Cheap insurance against fragmentation; does not change the peak requirement.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- preflight: does BS fit on THIS GPU? -------------------------------------
# Runs BEFORE the ~2.5 min dataset load. The alternative is what happened on
# 2026-08-28: three minutes of setup, then OOM at step 0 with nothing written.
python -u - <<PREFLIGHT
import sys, torch
PER_SAMPLE = 0.1786   # GiB/sample, from two measured points (OOM at 23.64, ran at 34)
FIXED      = 0.49     # GiB: weights + grads + AdamW moments
HEADROOM   = 1.0      # GiB for CUDA context, cuDNN workspace, fragmentation
bs     = ${BS}
total  = torch.cuda.get_device_properties(0).total_memory / 2**30
need   = FIXED + PER_SAMPLE * bs
budget = total - HEADROOM
print('preflight: %s %.2f GiB | bs %d needs ~%.2f GiB | budget %.2f GiB'
      % (torch.cuda.get_device_name(0), total, bs, need, budget))
if need > budget:
    safe = int((budget - FIXED) / PER_SAMPLE)
    sys.exit(
        'PREFLIGHT FAILED: bs %d does not fit on this GPU.\n'
        '  Largest bs that fits here: %d.\n'
        '  Either pin to a 32 GiB+ node (#SBATCH --nodelist=tikgpu05), or set\n'
        '  BS=%d -- and REPORT it. A different batch size is a different arm,\n'
        '  not a silent substitution: it changes steps/epoch and the step-count\n'
        '  claim in the write-up.' % (bs, safe, safe))
print('preflight OK')
PREFLIGHT

# --- stage the corpus --------------------------------------------------------
if [[ ! -d ${DATA_SRC}/neurons/train ]]; then
  echo "FATAL: no baked corpus at ${DATA_SRC}/neurons/train" >&2
  echo "       Checked node-local scratch and ${SHARE}/morphogen_npy." >&2
  echo "       If it is on another node's /scratch, pin this job there with --nodelist." >&2
  exit 1
fi
if [[ ${DATA_SRC} == /scratch/* ]]; then
  DATA=${DATA_SRC}
  echo "=== corpus already node-local, using in place: ${DATA} ==="
else
  DATA=${LOCAL}/data
  echo "=== staging the full corpus ${DATA_SRC} -> ${DATA} ==="
  mkdir -p "${DATA}"
  rsync -a "${DATA_SRC}/" "${DATA}/"
fi
echo "corpus: train $(ls ${DATA}/neurons/train/*.npy | wc -l), val $(ls ${DATA}/neurons/val/*.npy | wc -l)"

# --- resume from the newest checkpoint ---------------------------------------
# `epoch_%d.pth` is NOT lexicographically ordered (epoch_9 sorts after epoch_10),
# so pick by the numeric suffix or a long run silently restarts from epoch 9.
OUT=${RUNS}/${EXP}
RESUME=""
if compgen -G "${OUT}/epoch_*.pth" > /dev/null; then
  LATEST=$(ls "${OUT}"/epoch_*.pth | sed 's/.*epoch_\([0-9]*\)\.pth/\1 &/' | sort -n -k1,1 | tail -1 | cut -d' ' -f2)
  RESUME="--model ${LATEST}"
  echo "=== resuming from ${LATEST} ==="
else
  echo "=== no checkpoint in ${OUT}; starting fresh ==="
fi

# --- probe: measure throughput, then stop ------------------------------------
if [[ ${PROBE} -ne 0 ]]; then
  echo "=== PROBE: ${PROBE_EPOCHS} epochs at bs ${BS} to measure throughput ==="
  PROBE_OUT=${LOCAL}/probe.log
  # Probe writes to a scratch experiment dir so it cannot pollute the real run's
  # checkpoint grid, and saves nothing.
  python -u DDPM_train.py \
    --dataroot "${DATA}" --category neurons \
    --model_dir "${LOCAL}" --experiment_name probe \
    --bs ${BS} --niter ${PROBE_EPOCHS} --saveIter 100000 --use_ema \
    --workers ${SLURM_CPUS_PER_TASK:-8} \
    --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1 2>&1 | tee "${PROBE_OUT}"

  python -u - <<PY
import re, datetime
# The trainer logs "YYYY-MM-DD HH:MM:SS,mmm : [ e/ N][ i/ M]  loss: ..." per print_freq.
pat = re.compile(r'^(\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d),(\\d+) : \\[\\s*(\\d+)/\\s*\\d+\\]\\[\\s*(\\d+)/\\s*(\\d+)\\]')
rows = []
for line in open('${PROBE_OUT}'):
    m = pat.match(line)
    if m:
        t = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') + \
            datetime.timedelta(milliseconds=int(m.group(2)))
        rows.append((t, int(m.group(3)), int(m.group(4)), int(m.group(5))))
if len(rows) < 2:
    raise SystemExit('PROBE: not enough log lines to estimate; raise PROBE_EPOCHS')
steps_per_epoch = rows[0][3]
# Drop the first epoch: it carries CUDA context creation, cuDNN autotuning and
# dataloader worker spin-up, which do not recur.
warm = [r for r in rows if r[1] > 0] or rows
span = (warm[-1][0] - warm[0][0]).total_seconds()
nsteps = (warm[-1][1] - warm[0][1]) * steps_per_epoch + (warm[-1][2] - warm[0][2])
if span <= 0 or nsteps <= 0:
    raise SystemExit('PROBE: degenerate timing window; raise PROBE_EPOCHS')
sps = nsteps / span
total = ${NITER} * steps_per_epoch
print('\\n=== THROUGHPUT PROBE (bs ${BS}) ===')
print('  steps/epoch        : %d' % steps_per_epoch)
print('  measured           : %.3f s/step  (%.2f steps/s), warm epochs only' % (1/sps, sps))
print('  full run           : %d epochs x %d = %d steps' % (${NITER}, steps_per_epoch, total))
print('  projected          : %.1f GPU-hours  (%.1f days)' % (total/sps/3600, total/sps/86400))
print('  per 24h job        : %.0f epochs' % (86400*sps/steps_per_epoch))
print('\\n  dendrite_gen parity reference: 20,000 steps at batch_size 256')
PY
  echo "PROBE done -- no checkpoints written. Re-submit without PROBE=1 to train."
  echo "Finished at: $(date)"
  exit 0
fi

# --- train -------------------------------------------------------------------
# --saveIter 10 so the checkpoint grid is at least as fine as the selection grid
# (plan section 6). --use_ema because selection and generation both read EMA;
# EMA is now restored on resume, so a run split across jobs keeps its average.
mkdir -p "${OUT}"
echo "=== training: bs ${BS}, niter ${NITER}, saveIter ${SAVEITER} -> ${OUT} ==="
python -u DDPM_train.py \
  --dataroot "${DATA}" --category neurons \
  --model_dir "${RUNS}" --experiment_name ${EXP} \
  --bs ${BS} --niter ${NITER} --saveIter ${SAVEITER} --use_ema \
  --workers ${SLURM_CPUS_PER_TASK:-8} \
  --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1 \
  ${RESUME}

echo "=== checkpoints ==="
ls -la "${OUT}"/*.pth | tail -5
python -u - <<PY
import glob, re, torch
fs = sorted(glob.glob('${OUT}/epoch_*.pth'),
            key=lambda f: int(re.search(r'epoch_(\\d+)', f).group(1)))
print('%d checkpoints, epochs %s .. %s' % (
    len(fs),
    re.search(r'epoch_(\\d+)', fs[0]).group(1),
    re.search(r'epoch_(\\d+)', fs[-1]).group(1)))
k = torch.load(fs[-1], map_location='cpu').keys()
print('newest ->', sorted(k), '| ema present:', 'ema' in k)
if 'ema' not in k:
    raise SystemExit('FAILED: checkpoint has no ema; --use_ema did not take')
PY

echo "Finished at: $(date)"
exit 0

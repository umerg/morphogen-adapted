#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=mg_armS
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
# See scripts/train_armU.sbatch for the GPU-selection notes: --constraint matches
# node FEATURES and cannot pick a card on a mixed node; the mechanism that works is
# the GRES type, `sbatch --gres=gpu:<type>:1 scripts/train_armS.sbatch`. The
# preflight below is the backstop either way.

#
# Stage 6b -- Arm S: 690 neurons of ONE cell class at Arm P's exact step budget.
#
#   PROBE=1 sbatch scripts/train_armS.sbatch     # measure throughput, then exit
#   sbatch scripts/train_armS.sbatch             # the real run
#   sbatch scripts/train_armS.sbatch             # again -> auto-resumes
#
# Self-contained: it builds the subset on the node from the full bake, so there is
# nothing to prepare first.
#
# WHY THIS ARM EXISTS
#   Arm U epoch 299 has the right envelope -- gen_radius 1.08 against a training
#   distribution of exactly 1.0 -- but its clouds are not filamentary: local
#   linearity 0.605 against GT ~0.97, at ~16x the nearest-neighbour spacing. The
#   model learned where the neuron IS and not that its mass lies on thin branches.
#   Skeletonising a fuzzy cloud is what inflates node count to 2.66x GT, and no tau
#   fixes it: sweeping tau to 1.0 plateaus at 3.26x train GT, because the excess
#   nodes are not short TERMINAL branches and filter_short_branches cannot reach
#   anything else.
#
#   The paper (section 5.2) trains "batch size of 400 for 40,000 epochs on each neuron
#   category", on ~690-760 train neurons per model, three models. Gradient steps
#   are comparable to ours (~40-80k vs 60,900); PER-NEURON EXPOSURE differs 133x
#   (40,000 passes vs 300). For a DDPM that is the quantity that governs fine
#   detail: it learns eps(x_t, t) over 1000 timesteps and each neuron draws one
#   random t per epoch, so at 300 epochs a neuron is seen at under a third of
#   timesteps, and high-frequency structure lives at low t.
#
#   Arm S holds gradient steps EXACTLY equal to Arm P (60,900) and shrinks the
#   corpus to the paper's size, so per-neuron exposure rises ~34x and nothing else
#   about the budget moves.
#
#     Arm P   22,773 neurons, 7 classes    300 epochs   60,900 steps    300 passes/neuron
#     Arm S      690 neurons, 1 class   10,150 epochs   60,900 steps  10,150 passes/neuron
#            (NITER is derived at run time from the real steps/epoch, not hardcoded)
#     paper  ~690-760 neurons, 1 subtype 40,000 epochs  ~40-80k steps 40,000 passes/neuron
#
# CLASS 0 = 23P, the most populous (7,796 of 22,773 train), so 690 is a random
# sample of a large pool rather than nearly all of a small one. Names:
# dendrite_gen utils/data_loading.py:12 CELL_CLASS_NAMES.
#
# CONFOUND, for the write-up: this moves corpus size AND heterogeneity together.
# That is deliberate -- it is the paper's regime -- but a gain is therefore not
# attributable to exposure alone. The single-variable control is 690 neurons
# sampled across all 7 classes (CELL_CLASS=-1 is NOT implemented; that needs a
# different selector).
#
# READOUT is cloud-level and needs no reconstruction:
#   python tools/cloud_quality.py --clouds <gen>/clouds \
#       --ref-npy-root <subset>/neurons --ref-split val --train-split train
# Decide on linearity against the gt_baseline.json this script writes. If it moves
# materially toward GT, exposure is the driver and Arm L is justified. If it is flat
# at ~0.6, the blur is capacity or the bake and more training will not fix it.
#
# WALL CLOCK: same step count as Arm P, which took ~22.5 h -- inside --time=24:00:00
# but not by much, and 10,150 epoch boundaries is 34x Arm P's, with no
# persistent_workers. If the probe shows per-epoch overhead dominating, re-probe at
# WORKERS=0: the subset is a ~124 MB in-RAM array and __getitem__ is a slice. The
# TERM handler + resume covers an overrun either way.

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

# The 2026-08 bake carries cell_class=null in every sidecar (plan section 11 Stage 5),
# even though bake.sbatch passed --label-root -- the labels never reached them. They
# are not lost: they join by filename from the labelled SWC corpus, so this recovers
# them at subset-build time. No re-bake; the point clouds were never the problem.
LABEL_SRC=${LABEL_SRC:-${SHARE}/neurons_conditional_full}
CELL_CLASS=${CELL_CLASS:-0}          # 0 = 23P
N_TRAIN=${N_TRAIN:-690}              # paper: ~690-760 train neurons per model
SUBSET_SEED=${SUBSET_SEED:-0}
EXP=${EXP:-armS${N_TRAIN}_c${CELL_CLASS}_bs112}

BS=${BS:-112}
# NITER is DERIVED so the step budget matches Arm P exactly whatever the subset
# rounds to. Override NITER to force a different one (e.g. the paper-matched
# 40,000), or TARGET_STEPS to keep the derivation and change the target.
TARGET_STEPS=${TARGET_STEPS:-60900}
NITER=${NITER:-0}
# 350 divides the derived NITER (60900/6 = 10150 = 29 x 350) exactly, so the step
# budget stays exactly Arm P's AND the final epoch is checkpointed. 29 checkpoints,
# ~15 GB. A different BS changes steps/epoch and the rounding below takes over.
SAVEITER=${SAVEITER:-350}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
PROBE=${PROBE:-0}
PROBE_EPOCHS=${PROBE_EPOCHS:-40}     # NOT 3: see the probe block below

set -o errexit
set -o pipefail

if [[ -d /scratch && -w /scratch ]]; then
  LOCAL=/scratch/${ETH_USERNAME}/armS.${SLURM_JOB_ID:-$$}
else
  LOCAL=$(mktemp -d)/armS
fi
mkdir -p "${LOCAL}"
export TMPDIR="${LOCAL}"

# LOCAL holds only staging copies and the subset, never the source corpus, so it is
# always disposable. Checkpoints live on SHARE: a kill loses at most SAVEITER epochs
# and the next submission resumes. Guarded on LOCAL's shape rather than on DATA_SRC,
# which rm can never reach from here.
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- preflight: does BS fit on THIS GPU? -------------------------------------
# Before the dataset load, so a wrong card fails in seconds rather than after setup.
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
    sys.exit('PREFLIGHT FAILED: bs %d does not fit here. Largest that does: %d.\n'
             '  Pin to a 32 GiB+ card via the GRES type, or set BS=%d -- and REPORT\n'
             '  it: a different batch size changes steps/epoch and therefore the\n'
             '  derived NITER, so it is a different arm.' % (bs, safe, safe))
print('preflight OK')
PREFLIGHT

# --- locate the full bake ----------------------------------------------------
if [[ ! -d ${DATA_SRC}/neurons/train ]]; then
  echo "FATAL: no baked corpus at ${DATA_SRC}/neurons/train" >&2
  echo "       Checked node-local scratch and ${SHARE}/morphogen_npy." >&2
  echo "       If it is on another node's /scratch, pin this job there with --nodelist." >&2
  exit 1
fi
echo "=== full bake: ${DATA_SRC} ==="

# --- build the subset ON THIS NODE -------------------------------------------
# Node-local: 690 neurons is ~124 MB, so this is seconds, and it keeps the training
# read path off the share. The selection is seeded over a filename-sorted pool, so
# it is byte-identical on every node and every resubmission.
SUBSET=${LOCAL}/subset
echo "=== building the class-${CELL_CLASS} subset (n=${N_TRAIN}, seed=${SUBSET_SEED}) ==="
LABEL_ARG=""
if [[ -d ${LABEL_SRC} ]]; then
  LABEL_ARG="--label-root ${LABEL_SRC}"
else
  echo "WARNING: no labelled corpus at ${LABEL_SRC}; if the bake sidecars are" >&2
  echo "  unlabelled the build below will abort. Pass LABEL_SRC=<path>." >&2
fi
python -u tools/make_subset_synset.py \
  --npy-root "${DATA_SRC}" \
  --out-root "${SUBSET}" \
  ${LABEL_ARG} \
  --cell-class ${CELL_CLASS} \
  --n ${N_TRAIN} --seed ${SUBSET_SEED}

DATA=${SUBSET}
TRAIN_N=$(ls "${DATA}"/neurons/train/*.npy | wc -l)
VAL_N=$(ls "${DATA}"/neurons/val/*.npy | wc -l)
STEPS=$(( TRAIN_N / BS ))
if [[ ${STEPS} -lt 1 ]]; then
  # drop_last=True, so under one full batch per epoch the trainer spins through
  # every epoch doing nothing at all.
  echo "FATAL: ${TRAIN_N} train neurons is under one batch at bs ${BS} (drop_last)." >&2
  exit 1
fi
if [[ ${NITER} -le 0 ]]; then
  # Round to the nearest multiple of SAVEITER. DDPM_train saves when
  # (epoch+1) % saveIter == 0, so a NITER that is not a multiple trains its last
  # partial block with nothing written -- e.g. 10150 with saveIter 500 stops
  # checkpointing at epoch 9999 and then burns 150 more epochs for nothing.
  RAW_NITER=$(( TARGET_STEPS / STEPS ))
  NITER=$(( ( (RAW_NITER + SAVEITER / 2) / SAVEITER ) * SAVEITER ))
  [[ ${NITER} -lt ${SAVEITER} ]] && NITER=${SAVEITER}
fi
LAST_EPOCH=$(( ( NITER / SAVEITER ) * SAVEITER - 1 ))

OUT=${RUNS}/${EXP}
mkdir -p "${OUT}"

# The subset is rebuilt every job, so a resumed run MUST get the same 690 neurons.
# Selection is deterministic given the same source, but the source is not immutable
# -- a re-bake or an added file would silently change the training set mid-run and
# nothing downstream would show it. Pin it on the first job and compare thereafter.
if [[ -f ${OUT}/subset.json ]]; then
  python -u - <<PY
import json, sys
ref = json.load(open('${OUT}/subset.json'))
new = json.load(open('${SUBSET}/subset.json'))
if ref['files'] != new['files']:
    r, n = ref['files']['train'], new['files']['train']
    sys.exit(
        'FATAL: the subset rebuilt on this node differs from the one this run '
        'started with.\n'
        '  recorded %d train / %d val, rebuilt %d train / %d val; %d train files differ.\n'
        '  The source bake changed under the run. Resuming would train on different\n'
        '  data than the existing checkpoints saw. Investigate ${DATA_SRC}, or start\n'
        '  a fresh EXP.' % (len(r), len(ref['files']['val']), len(n),
                            len(new['files']['val']), len(set(r) ^ set(n)) ))
print('subset matches the one recorded for this run (%d train / %d val)'
      % (len(new['files']['train']), len(new['files']['val'])))
PY
else
  cp "${SUBSET}/subset.json" "${OUT}/subset.json"
  echo "recorded the subset selection -> ${OUT}/subset.json"
fi

echo
echo "=== budget ==="
echo "  ${TRAIN_N} train / ${VAL_N} val, class ${CELL_CLASS}"
echo "  ${TRAIN_N} / bs ${BS} = ${STEPS} steps/epoch (drop_last)"
echo "  NITER=${NITER} x ${STEPS} = $(( NITER * STEPS )) steps   (target ${TARGET_STEPS})"
if [[ $(( NITER * STEPS )) -ne ${TARGET_STEPS} ]]; then
  echo "  NOTE: NITER was rounded to a multiple of SAVEITER=${SAVEITER} so the final"
  echo "        epoch is checkpointed; $(( NITER * STEPS )) vs ${TARGET_STEPS} steps."
  echo "        REPORT the actual number, not the target."
fi
echo "  per-neuron exposure: ${NITER} passes   (Arm P: 300, paper: 40,000)"
echo "  last checkpoint will be epoch_${LAST_EPOCH}.pth"

# --- GT cloud baseline, once -------------------------------------------------
# The number Arm S is judged against, and it must come from THIS bake. A value taken
# from a cleaned-corpus bake is inflated: interpolated straight segments are
# artificially filamentary, which would overstate the gap.
if [[ ! -f ${OUT}/gt_baseline.json ]]; then
  echo
  echo "=== GT cloud baseline (class ${CELL_CLASS} val, up to 200 neurons) ==="
  python -u tools/cloud_quality.py \
    --ref-npy-root "${DATA}/neurons" --ref-split val --limit 200 \
    --output-json "${OUT}/gt_baseline.json"
fi

# --- resume from the newest checkpoint ---------------------------------------
# `epoch_%d.pth` is NOT lexicographically ordered (epoch_9 sorts after epoch_10),
# so pick by the numeric suffix or a long run silently restarts from epoch 9.
RESUME=""
if compgen -G "${OUT}/epoch_*.pth" > /dev/null; then
  LATEST=$(ls "${OUT}"/epoch_*.pth | sed 's/.*epoch_\([0-9]*\)\.pth/\1 &/' | sort -n -k1,1 | tail -1 | cut -d' ' -f2)
  RESUME="--model ${LATEST}"
  echo "=== resuming from ${LATEST} ==="
else
  echo "=== no checkpoint in ${OUT}; starting fresh ==="
fi

# --- probe: measure throughput, then stop ------------------------------------
# PROBE_EPOCHS defaults to 40 here, not Arm U's 3. --print_freq is 50 and `i` is the
# WITHIN-EPOCH index (DDPM_train.py:725), so at ~6 steps/epoch the trainer emits
# exactly one log line per epoch; 3 epochs leaves the estimator 2 rows. 40 also spans
# enough epoch boundaries to price the per-epoch dataloader worker respawn, which is
# the main schedule risk at 10,150 epochs.
if [[ ${PROBE} -ne 0 ]]; then
  echo "=== PROBE: ${PROBE_EPOCHS} epochs at bs ${BS} ==="
  PROBE_OUT=${LOCAL}/probe.log
  python -u DDPM_train.py \
    --dataroot "${DATA}" --category neurons \
    --model_dir "${LOCAL}" --experiment_name probe \
    --bs ${BS} --niter ${PROBE_EPOCHS} --saveIter 100000 --use_ema \
    --workers ${WORKERS} \
    --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1 2>&1 | tee "${PROBE_OUT}"
  python -u - <<PY
import re, datetime
pat = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) : \[\s*(\d+)/\s*\d+\]\[\s*(\d+)/\s*(\d+)\]')
rows = []
for line in open('${PROBE_OUT}'):
    m = pat.match(line)
    if m:
        t = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') + \
            datetime.timedelta(milliseconds=int(m.group(2)))
        rows.append((t, int(m.group(3)), int(m.group(4)), int(m.group(5))))
if len(rows) < 2:
    raise SystemExit('PROBE: not enough log lines; raise PROBE_EPOCHS')
steps_per_epoch = rows[0][3]
warm = [r for r in rows if r[1] > 0] or rows
span = (warm[-1][0] - warm[0][0]).total_seconds()
nsteps = (warm[-1][1] - warm[0][1]) * steps_per_epoch + (warm[-1][2] - warm[0][2])
if span <= 0 or nsteps <= 0:
    raise SystemExit('PROBE: degenerate timing window; raise PROBE_EPOCHS')
sps = nsteps / span
total = ${NITER} * steps_per_epoch
print('\n=== THROUGHPUT PROBE (bs ${BS}, ${TRAIN_N} neurons) ===')
print('  steps/epoch        : %d  (expected ${STEPS})' % steps_per_epoch)
print('  measured           : %.3f s/step  (%.2f steps/s), warm epochs only' % (1/sps, sps))
print('  full run           : ${NITER} epochs x %d = %d steps' % (steps_per_epoch, total))
print('  projected          : %.1f GPU-hours  (%.1f days)' % (total/sps/3600, total/sps/86400))
print('  per 24h job        : %.0f epochs' % (86400*sps/steps_per_epoch))
print('\n  Arm P reference: 60,900 steps in ~22.5 h. Much slower here means per-epoch')
print('  overhead (10,150 dataloader worker respawns) -- re-probe with WORKERS=0.')
PY
  echo "PROBE done -- no checkpoints written. Re-submit without PROBE=1 to train."
  echo "Finished at: $(date)"
  exit 0
fi

# --- train -------------------------------------------------------------------
echo
echo "=== training: bs ${BS}, niter ${NITER}, saveIter ${SAVEITER} -> ${OUT} ==="
python -u DDPM_train.py \
  --dataroot "${DATA}" --category neurons \
  --model_dir "${RUNS}" --experiment_name ${EXP} \
  --bs ${BS} --niter ${NITER} --saveIter ${SAVEITER} --use_ema \
  --workers ${WORKERS} \
  --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1 \
  ${RESUME}

echo "=== checkpoints ==="
ls -la "${OUT}"/*.pth | tail -5
python -u - <<PY
import glob, re, torch
fs = sorted(glob.glob('${OUT}/epoch_*.pth'),
            key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
print('%d checkpoints, epochs %s .. %s' % (
    len(fs), re.search(r'epoch_(\d+)', fs[0]).group(1),
    re.search(r'epoch_(\d+)', fs[-1]).group(1)))
k = torch.load(fs[-1], map_location='cpu').keys()
print('newest ->', sorted(k), '| ema present:', 'ema' in k)
if 'ema' not in k:
    raise SystemExit('FAILED: checkpoint has no ema; --use_ema did not take')
PY

# The subset lives on node-local scratch and dies with this job, so say how to get it
# back -- generation needs the SAME corpus (its standardisation constants come from
# this train split, and gt_scale_m is joined from these sidecars).
cat <<NEXT

=== next: sample and score, no reconstruction needed ===
  # rebuild the identical subset (same flags -> same 690 neurons):
  python tools/make_subset_synset.py --npy-root ${DATA_SRC} \\
      --label-root ${LABEL_SRC} --out-root \$SUBSET \\
      --cell-class ${CELL_CLASS} --n ${N_TRAIN} --seed ${SUBSET_SEED}
  python -u morphology_gen.py --stage sample --max_samples 150 \\
      --dataroot \$SUBSET --category neurons \\
      --model ${OUT}/epoch_${LAST_EPOCH}.pth \\
      --generate_dir ${SHARE}/mg_gen/${EXP} --bs 50 --num_classes 1
  python tools/cloud_quality.py --clouds ${SHARE}/mg_gen/${EXP}/clouds \\
      --ref-npy-root \$SUBSET/neurons --ref-split val --train-split train \\
      --output-json ${SHARE}/mg_gen/${EXP}/cq.json
  # compare linearity against ${OUT}/gt_baseline.json
NEXT

echo "Finished at: $(date)"
exit 0

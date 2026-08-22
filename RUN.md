# Running MorphoGen as a dendrite_gen baseline (Linux + CUDA)

Source of truth for environment, data staging, and the exact commands for the dry run and the
main runs. Companion to the plan in `~/.claude/plans/for-now-assume-tat-federated-locket.md`.

**Why not on the Mac.** Measured: DiT-S/4 trains at **7.8 samples/s on CPU** here. The 200-neuron
dry run would take ~9 min, but the main run (22,773 neurons x 300 epochs = 6.8M sample-visits)
would take **~10 days**. MPS fails outright (device-mismatch inside the model). More importantly
the dry run exists to de-risk the *CUDA* path — running it on CPU would exercise none of what
actually breaks. Run both on the cluster.

---

## 0. Data paths

Every path is a CLI flag with an environment-variable default, so you set them once per shell
rather than repeating them. Nothing falls back to a hardcoded location: an unset or wrong
`--dataroot` **exits immediately** with a message rather than yielding an empty dataset and a
training loop that appears to run while doing nothing.

```bash
export MORPHOGEN_DATAROOT=/scratch/$USER/morphogen_npy   # <root>/<synset>/{train,val,test}/*.npy
export MORPHOGEN_RUNS=/scratch/$USER/mg_runs             # checkpoints + logs
export MORPHOGEN_GENDIR=/scratch/$USER/mg_gen            # generated SWCs + manifest.json
export MORPHOGEN_CATEGORY=neurons                        # or class_0,...,class_6
export DENDRITE_GEN=/scratch/$USER/dendrite_gen          # only needed by tools/recon_ref.py
```

| variable | flag | default if unset |
|---|---|---|
| `MORPHOGEN_DATAROOT` | `--dataroot` | *(empty -> exits with an error)* |
| `MORPHOGEN_CATEGORY` | `--category` | `neurons` |
| `MORPHOGEN_RUNS` | `--model_dir` | `./runs` |
| `MORPHOGEN_GENDIR` | `--generate_dir` | `./generated` |
| `DENDRITE_GEN` | — | `~/Documents/dendrite_gen` |

An explicit flag always wins over the environment. The `tools/` scripts take their paths as
required arguments (`--raw-root`, `--label-root`, `--out-root`, `--npy-root`) — only
`DENDRITE_GEN` is environment-driven there.

With these exported, the commands below shorten to e.g.
`python DDPM_train.py --experiment_name uncond_parity --bs 256 --niter 300 --saveIter 10 --use_ema`.

---

## 1. One-time setup

**Python must be >= 3.10.** A 3.8 env caps torch at 2.4.1 and numpy at 1.24 and cannot satisfy
`requirements-baseline.txt`. The upstream README's `python==3.8.5` describes the authors' 2023
environment, not this fork.

```bash
git clone <this repo> && cd MorphoGen
git checkout baseline-dendrite-gen

conda create -n MORPHOGEN python=3.10 -y
conda activate MORPHOGEN

# 1. torch FIRST, matching the driver. Read the CUDA version from nvidia-smi.
nvidia-smi | head -3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. everything else (this file does NOT pin torch, so it cannot clobber step 1)
pip install -r requirements-baseline.txt
```

**Order matters.** A cu121 build runs on any 12.x driver. `torch >= 2.6` dropped cu121 wheels, so
on a 12.0-12.3 driver pin `torch==2.5.1` + `torchvision==0.20.1`. Installing torch from the
default index instead gives a build compiled against a newer CUDA and fails at `.cuda()` with:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12020)
```

which means the *torch build* is too new for the driver, not that the driver needs updating.
Check with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

The shipped `requirements.txt` is a raw `pip freeze` with ~15 unusable local
`@ file:///croot/...` paths and cannot be installed — use `requirements-baseline.txt`.

**No CUDA extension build is required.** `modules/functional/src` is absent from the release, so
`modules/functional/{voxelization,devoxelization}.py` provide pure-PyTorch equivalents of the two
ops DiT-3D actually uses, and `backend.py` degrades gracefully. Expect a one-time
`RuntimeWarning: PVCNN CUDA backend unavailable` — that is normal and correct. To use the faster
compiled kernels, vendor upstream `src/` from https://github.com/DiT-3D/DiT-3D; the loader picks
it up automatically. Verify equivalence before trusting it (`allclose` at 1e-5).

Sanity check:
```bash
python -c "
import torch; from models.dit3d import DiT3D_models
m=DiT3D_models['DiT-S/4'](pretrained=False,input_size=32,num_classes=7).cuda()
x=torch.randn(2,3,2048).cuda(); t=torch.randint(0,1000,(2,)).cuda(); y=torch.randint(0,7,(2,)).cuda()
print('forward OK', tuple(m(x,t,y).shape), '| params %.1fM'%(sum(p.numel() for p in m.parameters())/1e6))"
```
Expect `forward OK (2, 3, 2048) | params 32.8M`.

---

## 2. Stage the data

Two corpora are needed. `neurons_raw` has the degree-2 (continuation) nodes MorphoGen requires;
`neurons_conditional_full` supplies the `# cell_class` labels, joined by filename.

```bash
rsync -a ~/Documents/neurons_raw/                <cluster>:/scratch/guptau/neurons_raw/
rsync -a ~/Documents/neurons_conditional_full/   <cluster>:/scratch/guptau/neurons_conditional_full/
```

Bake the point clouds (CPU, parallel, ~15 min for the full corpus at 16 workers):

```bash
# unconditional arm -- single synset "neurons"
python tools/swc_to_morphogen_npy.py \
  --raw-root   /scratch/guptau/neurons_raw \
  --label-root /scratch/guptau/neurons_conditional_full \
  --out-root   /scratch/guptau/morphogen_npy \
  --mode uncond --workers 16

# class-conditional arm -- one synset per cell class (class_0 .. class_6)
python tools/swc_to_morphogen_npy.py \
  --raw-root   /scratch/guptau/neurons_raw \
  --label-root /scratch/guptau/neurons_conditional_full \
  --out-root   /scratch/guptau/morphogen_npy_cls \
  --mode class --workers 16
```

Each `.npy` is exactly `(15000, 3)` (the loader asserts this) with a `.meta.json` sidecar holding
`centroid`, `scale_m` and `cell_class`. **The sidecars are load-bearing** — `pc_normlize` puts
every neuron on the unit sphere, so without them absolute scale is unrecoverable and every
scale-dependent metric is meaningless.

Verify:
```bash
python - <<'EOF'
import glob, numpy as np
for split in ['train','val','test']:
    f=sorted(glob.glob(f'/scratch/guptau/morphogen_npy/neurons/{split}/*.npy'))
    a=np.load(f[0]); print(split, len(f), a.shape, a.dtype, 'finite', bool(np.isfinite(a).all()))
EOF
```
Expect 22773 / 2529 / 1167 and `(15000, 3) float32`.

---

## 3. Dry run — GATE. Do not request real hours before this is green.

Exercises every interface on ~200 neurons. The point is interfaces, not numbers.

```bash
export CUDA_VISIBLE_DEVICES=0        # upstream hardcoded GPU 2; now env-driven

python DDPM_train.py \
  --dataroot /scratch/guptau/morphogen_npy \
  --category neurons \
  --model_dir /scratch/guptau/mg_runs --experiment_name dryrun \
  --bs 64 --niter 20 --saveIter 5 --use_ema \
  --npoints 2048 --model_type DiT-S/4 --voxel_size 32 --num_classes 1
```

**`--bs` must be <= the number of training neurons.** The train loader uses `drop_last=True`, so
`--bs 256` against a 200-neuron subset yields **zero** batches and the loop silently does nothing.

Then generation + reconstruction:

```bash
python morphology_gen.py \
  --dataroot /scratch/guptau/morphogen_npy --category neurons \
  --model /scratch/guptau/mg_runs/dryrun/epoch_19.pth \
  --generate_dir /scratch/guptau/mg_gen/dryrun \
  --bs 16 --num_classes 1 \
  --detect_radius 0.30 --root_cap 23 --gamma_seed 0.40 --length_threshold 0.30
```

Checklist — all must pass:
- [ ] training completes; `epoch_{4,9,14,19}.pth` written, each containing an `ema` key
- [ ] generation writes one `<mid>.swc` per val neuron **plus `manifest.json`**
- [ ] the manifest is a bijection onto the val split (never trust positional pairing: the dataset
      applies a fixed `random.Random(38383)` shuffle)
- [ ] `tools/recon_ref.py` runs on the generated SWCs without a "not a tree" assertion

---

## 4. Main runs

Two conditioning arms x two budgets. `--gamma_seed`/`--length_threshold` are **reconstruction**
parameters and affect generation only, not training.

```bash
# Arm U-P: unconditional, parity budget (E=300, bs 256 -> ~26.7k steps)
python DDPM_train.py --dataroot /scratch/guptau/morphogen_npy --category neurons \
  --model_dir /scratch/guptau/mg_runs --experiment_name uncond_parity \
  --bs 256 --niter 300 --saveIter 10 --use_ema --num_classes 1

# Arm C-P: class-conditional, same budget
python DDPM_train.py --dataroot /scratch/guptau/morphogen_npy_cls \
  --category class_0,class_1,class_2,class_3,class_4,class_5,class_6 \
  --model_dir /scratch/guptau/mg_runs --experiment_name cond_parity \
  --bs 256 --niter 300 --saveIter 10 --use_ema --num_classes 7
```

`--bs 256` matches dendrite_gen's `training.batch_size: 256` and lands within ~35% of its 20k
gradient steps, so the two methods match on batch size *and* step count simultaneously — a
stronger parity claim than epoch-matching alone (`docs/BUDGET_PARITY_SEMLAFLOW.md` matches epochs
and reports steps).

**Arm L (best-effort):** rerun with `--niter 1500` and pick the checkpoint where the 1-NN-CD
curve flattens. If it beats the parity arm, report it as the baseline's headline — giving the
baseline its best shot is what makes the comparison defensible.

Report **measured GPU-hours**, split `train only` / `to best checkpoint`, excluding validation
time on both sides (`docs/BUDGET_PARITY_SEMLAFLOW.md` section 5.4).

### Checkpoint selection

Upstream has **no validation at all** — it saves every `--saveIter` epochs and never tracks a
best. Select on **1-NN-CD** over generated-vs-val point clouds (DDIM-100, 512 samples from EMA,
clouds subsampled to 512 points for the pairwise, fixed noise seed across epochs so the curve is
comparable). `metrics/evaluation_metrics.py:11 distChamfer` is pure torch — call
`_pairwise_EMD_CD_(..., accelerated_cd=False)`; no CUDA extension needed, and EMD is not used.

Generate from **EMA** weights. `morphology_gen.py` loads `model_state`, so patch it to prefer
`ema` or you will select on EMA and generate from raw weights.

---

## 5. Reconstruction parameters (calibrated on TRAIN, confirmed on held-out val)

| arm | flags |
|---|---|
| **faithful** (primary, as published) | `--detect_radius 0.30 --root_cap 23` |
| **MorphoGen+** (documented deviation) | `--detect_radius 0.30 --root_cap 23 --gamma_seed 0.40 --length_threshold 0.30` |

Defaults are the shipped values, so the faithful arm runs unless the flags are passed.
On 300 held-out val neurons, MorphoGen+ improves mean normalised W1 from **2.51 to 0.61 (4.1x)**.
`--detect_radius` is mandatory in both arms: the shipped 5.0 exceeds the entire extent of
unit-sphere data, so every local density ties and the soma becomes an arbitrary point.

Generation and reconstruction should be run as **two separate resumable jobs** — persist the
point clouds first (cheap, GPU), then reconstruct on CPU (~4 s/neuron, embarrassingly parallel,
~5 min for val across 32 workers). Upstream fuses them and then crashes after the expensive loop.

---

## 6. Known upstream bugs already fixed on this branch

Kept here so cluster-side surprises are recognisable. Full rationale in the plan's deviations
table; each is either a bug fix or a restoration of behaviour the paper describes.

| file | issue |
|---|---|
| `modules/functional/backend.py` | `src/` absent -> `import DDPM_train` failed on every machine |
| `DDPM_train.py:17` | hardcoded `CUDA_VISIBLE_DEVICES="2"` |
| `DDPM_train.py` `--bs` | no `type=int`; CLI overrides arrived as strings |
| `sub_process.py` `getBranch` | 1-based id assumption corrupted every branch on the 0-indexed corpus (fabricated segments spanning the whole neuron) |
| `sub_process.py` FPS | `np.compat.long` removed in numpy 2.x -> crashed outright |
| `utils/ske_connect.py` `L1_medial` | seeded stdlib RNG but drew centres from the numpy global RNG -> irreproducible run-to-run |
| `utils/ske_connect.py` `fps` | unseeded start index, and bounded by `npoint` instead of `N` |
| `utils/cut.py` | `filter_short_branches` could delete the root |
| `utils/swc_denoise.py:250` | `parse_args()` on `sys.argv` mid-generation -> `SystemExit(2)` |
| `morphology_gen.py:688` | `return output_dir` NameError *after* the full generation loop |
| `morphology_gen.py` aux ckpt | pointed at `./temp/resnet16model.pth`; shipped file is `trained_model/Auxiliary.pth` |
| `morphology_gen.py` output naming | `pc_{counter}.swc` against a shuffled loader -> silently mispaired every paired metric |
| `datasets/shapenet_data_pc.py` | neuron synsets were unregistered -> loader found no data |

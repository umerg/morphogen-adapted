#!/bin/bash
# Where did the baked point clouds actually go?  Run on the login node.
#   bash scripts/verify_bake.sh [out_root]
OUT=${1:-/scratch/guptau/morphogen_npy}
SCRATCH=/scratch/guptau

echo "=== 1. counts at the expected destination: ${OUT} ==="
for s in train val test; do
  n=$(find "${OUT}/neurons/${s}" -name '*.npy' 2>/dev/null | wc -l)
  m=$(find "${OUT}/neurons/${s}" -name '*.meta.json' 2>/dev/null | wc -l)
  printf '  %-6s %6s npy  %6s meta\n' "$s" "$n" "$m"
done
echo "  want:  train 22773 / val 2529 / test 1167 = 26469"
du -sh "${OUT}" 2>/dev/null | sed 's/^/  size: /'

echo
echo "=== 2. is a sample file actually valid? ==="
python - "$OUT" <<'PY'
import glob, json, sys
try:
    import numpy as np
except ImportError:
    sys.exit('  (numpy not on PATH -- run inside the MOPRHO2 env for this check)')
f = sorted(glob.glob(f'{sys.argv[1]}/neurons/*/*.npy'))
if not f:
    sys.exit('  no .npy found at all')
for p in (f[0], f[len(f)//2], f[-1]):
    a = np.load(p)
    try:
        m = json.load(open(p.replace('.npy', '.meta.json')))
        s = f"scale_m={m['scale_m']:.1f} class={m['cell_class']}"
    except Exception as e:
        s = f'SIDECAR BAD: {e}'
    print(f'  {p.split("/")[-1]:40} {a.shape} {a.dtype} '
          f'{"finite" if np.isfinite(a).all() else "NONFINITE"}  {s}')
PY

echo
echo "=== 3. did they land somewhere else on the share? ==="
find "${SCRATCH}" -maxdepth 4 -name '*.npy' -newermt '2026-08-22' -printf '%h\n' 2>/dev/null \
  | sort | uniq -c | sort -rn | head
echo

echo "=== 4. recent jobs -- which node ran it, and for how long ==="
sacct -u "$USER" -S 2026-08-22 -o JobID%12,JobName%14,NodeList%12,Elapsed,State,ExitCode 2>/dev/null | head -20

echo
echo "=== 5. is the output still sitting on that node's local disk? ==="
nodes=$(sacct -u "$USER" -S 2026-08-22 -o NodeList%20 -n 2>/dev/null | tr -d ' ' | grep -E '^[a-z]' | sort -u)
for n in $nodes; do
  echo "  --- $n ---"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$n" \
    'ls -d /scratch/'"$USER"'/* /scratch/*'"$USER"'* /tmp/tmp.* 2>/dev/null | head;
     find /scratch -maxdepth 4 -name "*.npy" 2>/dev/null | wc -l | sed "s/^/  npy on local scratch: /"' \
    2>&1 | sed 's/^/    /'
done

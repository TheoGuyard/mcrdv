#!/bin/bash
# Generic array job for one mcrdv experiment on the Digital Research Alliance
# (Compute Canada) clusters. Submit it through ../submit.sh:
#
#     SWEEP=$(./sweeps/scaling.py --quiet)
#     ./submit.sh scaling "$SWEEP" --array 0-9
#
# ---------------------------------------------------------------------------
# SITE SETTINGS -- check these once against your own allocation.
#   --account     required everywhere; ask `sshare -U` if you do not know it.
#   --time        a task runs its whole share of the configurations one after
#                 another, so size this as
#                   configs_per_task x (search budget + setup) + ~15 s to compile
#   --cpus-per-task  only the bound table is threaded; see the note at the end.
# ---------------------------------------------------------------------------
#SBATCH --account=def-CHANGEME
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=mcrdv
#SBATCH --output=%x-%A_%a.out

set -euo pipefail

: "${EXPERIMENT:?set EXPERIMENT, or submit through submit.sh}"
: "${PROJECT:?set PROJECT to the repository root}"
: "${CONFIGS:?set CONFIGS to a campaign directory, or submit through submit.sh}"
OUT="${OUT:-$PROJECT/experiments/$EXPERIMENT/results}"

module load StdEnv/2023 python/3.11

# Built per job on node-local storage rather than kept on the shared
# filesystem: a virtualenv is thousands of small files, which is the access
# pattern Lustre handles worst, and $SLURM_TMPDIR is wiped for us afterwards.
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy numba pyyaml
pip install --no-index --no-deps -e "$PROJECT"

# numba writes its on-disk cache here. Node-local for the same reason, and it
# only ever holds the array-only helpers -- anything typed on a layer kernel is
# recompiled per process regardless.
export NUMBA_CACHE_DIR="$SLURM_TMPDIR/numba"

mkdir -p "$OUT"

python "$PROJECT/experiments/$EXPERIMENT/run.py" "$CONFIGS" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --num-tasks "$SLURM_ARRAY_TASK_COUNT" \
    --out "$OUT"

# ---------------------------------------------------------------------------
# Note on --cpus-per-task. The only parallel section is the table of per-leg
# bounds, filled row-wise over a thread pool in `mcrdv.sequence.hgs.pair_lb`.
# It is built with the default pool size, which is derived from os.cpu_count()
# -- the whole node, not this task's allocation -- so a small --cpus-per-task
# oversubscribes and makes the setup timing noisy. Until `workers` is plumbed
# through HgsSequence, either request a large share of the node or treat the
# reported t_setup on the biggest catalogue as an upper bound.
# ---------------------------------------------------------------------------

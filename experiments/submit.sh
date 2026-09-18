#!/bin/bash
# Submit one experiment's campaign as a Slurm array job.
#
#     SWEEP=$(./sweeps/scaling.py --quiet)
#     ./submit.sh scaling "$SWEEP" --array 0-9
#     ./submit.sh sensitivity "$SWEEP" --array 0-19%5 --time 06:00:00
#
# Everything after the campaign directory is passed straight to sbatch, so any
# directive in slurm/array.sbatch can be overridden per submission.
set -euo pipefail

if [[ $# -lt 2 || "$1" == -* ]]; then
    echo "usage: $0 <experiment> <campaign-dir> [sbatch options...]" >&2
    echo "  e.g. $0 scaling \"\$(./sweeps/scaling.py --quiet)\" --array 0-9" >&2
    exit 2
fi

EXPERIMENT="$1"
CONFIGS="$2"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

if [[ ! -f "$HERE/$EXPERIMENT/run.py" ]]; then
    echo "no experiment '$EXPERIMENT' in $HERE" >&2
    echo "available:" >&2
    for dir in "$HERE"/*/; do
        [[ -f "$dir/run.py" ]] && echo "  $(basename "$dir")" >&2
    done
    exit 2
fi

if [[ ! -d "$CONFIGS" ]]; then
    echo "campaign directory '$CONFIGS' does not exist" >&2
    echo "generate one first:  ./sweeps/$EXPERIMENT.py" >&2
    exit 2
fi

# How many configurations there are decides what array size makes sense, and a
# campaign that is empty or far bigger than expected is much cheaper to notice
# here than in a job that starts an hour from now.
COUNT=$(find "$CONFIGS" -name 'run-*.yaml' | wc -l | tr -d ' ')
if [[ "$COUNT" -eq 0 ]]; then
    echo "no configurations in $CONFIGS" >&2
    exit 2
fi
echo "$EXPERIMENT: $COUNT configurations in $CONFIGS"

exec sbatch \
    --export=ALL,EXPERIMENT="$EXPERIMENT",PROJECT="$PROJECT",CONFIGS="$CONFIGS" \
    --job-name="mcrdv-$EXPERIMENT" \
    "$@" \
    "$HERE/slurm/array.sbatch"

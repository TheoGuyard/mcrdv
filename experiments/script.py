#!/usr/bin/env python3
"""Scripts for running the experiments on the cluster.
    ./script.py send                # send local workspace to remote cluster
    ./script.py install             # build the environment there
    ./script.py make                # write every campaign and its job file
    ./script.py run -s <name>       # submit one campaign as an array job
    ./script.py receive -s <name>   # retrieve locally remote results on cluster
    ./script.py clean               # clear the generated campaigns
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import textwrap
from typing import Any, Dict, Iterator, List, Mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import config as configuration  # noqa: E402
from _common import generate  # noqa: E402
from _common import record as recording  # noqa: E402

# ----- Global variables ----- #

LOC_PATH = "~/Documents/Github/mcrdv"
REM_PATH = "tguyard@rorqual.alliancecan.ca:links/scratch"
HOME_DIR = "/home/tguyard"
VENV_DIR = "/home/tguyard/.venv"
LOGS_DIR = "/home/tguyard/logs"
ACCOUNT = "def-vidalthi"
MODULES = ["StdEnv/2023", "python/3.11"]
REQUIRES = ["numpy", "numba", "pyyaml"]

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
EXPS_DIR = BASE_DIR / "experiments"
TMPS_DIR = EXPS_DIR / "tmp"
SWEEPS_PATH = EXPS_DIR / "sweeps.yaml"
JOB_NAME = "job.sh"


# ----- Sweep management ----- #


def sweeps() -> Dict[str, Dict[str, Any]]:
    """Every sweep of `sweeps.yaml`, with the defaults filled in."""
    if not SWEEPS_PATH.exists():
        raise SystemExit(f"{SWEEPS_PATH} does not exist")
    document = configuration.load(SWEEPS_PATH)
    defaults = document.get("defaults") or {}
    out = {}
    for name, sweep in (document.get("sweeps") or {}).items():
        merged = {**defaults, **(sweep or {})}
        merged.setdefault("experiment", name)
        out[name] = merged
    return out


def pick(name: str | None) -> Dict[str, Dict[str, Any]]:
    """Every sweep, or the one named."""
    known = sweeps()
    if name is None:
        return known
    if name not in known:
        raise SystemExit(f"unknown sweep '{name}'")
    return {name: known[name]}


def values(spec: Any) -> List[Any]:
    """One axis's values: a list as written, or a `range` shorthand."""
    if isinstance(spec, Mapping):
        if set(spec) != {"range"}:
            raise SystemExit(f"an axis is a list or a range, got {spec!r}")
        bounds = spec["range"]
        bounds = [bounds] if isinstance(bounds, int) else list(bounds)
        return list(range(*bounds))
    if not isinstance(spec, list):
        raise SystemExit(f"an axis is a list or a range, got {spec!r}")
    return list(spec)


def configs(base: Mapping[str, Any], sweep: Mapping[str, Any]) -> Iterator:
    """Every configuration a sweep campaign describes."""
    fixed = sweep.get("set") or {}
    if not isinstance(fixed, Mapping):
        raise SystemExit(f"a sweep's 'set' is a mapping, got {fixed!r}")
    base = configuration.derive(base, **fixed)

    for arm in sweep.get("campaign") or []:
        axes = dict((arm or {}).get("axes") or {})
        bundles = [{}]
        if "set" in axes:
            bundles = [dict(bundle) for bundle in axes.pop("set")]
        axes = {path: values(spec) for path, spec in axes.items()}
        mode = str((arm or {}).get("mode", "grid"))

        for bundle in bundles:
            start = configuration.derive(base, **bundle)
            if mode == "grid":
                yield from generate.grid(start, **axes)
            elif mode == "one_at_a_time":
                replicates = axes.pop("seed", [0])
                yield from generate.one_at_a_time(
                    start, seed=replicates, **axes
                )
            else:
                raise SystemExit(
                    f"unknown mode '{mode}' (known: grid, one_at_a_time)"
                )


# ----- Slurm files ----- #


def slurm_job_stream(name: str, sweep: Mapping[str, Any], count: int) -> str:
    """The array job that runs one campaign."""
    experiment = sweep["experiment"]
    tasks = min(int(sweep["tasks"]), max(count, 1))
    return textwrap.dedent(
        """\
        #!/bin/sh
        #SBATCH -J mcrdv-{}
        #SBATCH -o {}/%x_%A_%a.out
        #SBATCH -e {}/%x_%A_%a.err
        #SBATCH -t {}
        #SBATCH --mem={}
        #SBATCH --cpus-per-task={}
        #SBATCH --array=0-{}
        #SBATCH --account={}

        module load {}
        source {}/.bash_profile
        source {}/.bashrc
        source {}/bin/activate

        export NUMBA_CACHE_DIR=$SLURM_TMPDIR/numba

        cd {}
        python experiments/{}/run.py {}/{}/configs \\
            --task-id $SLURM_ARRAY_TASK_ID \\
            --num-tasks $SLURM_ARRAY_TASK_COUNT
    """.format(
            name,
            LOGS_DIR,
            LOGS_DIR,
            sweep["walltime"],
            sweep["memory"],
            sweep["cpus"],
            tasks - 1,
            sweep["account"],
            " ".join(MODULES),
            HOME_DIR,
            HOME_DIR,
            VENV_DIR,
            BASE_DIR,
            experiment,
            TMPS_DIR,
            name,
        )
    )


# ----- Command line functions ----- #


def send() -> None:
    """Send the codebase to the cluster."""
    print("send")
    command = " ".join(
        [
            "rsync -amv",
            "--exclude '.git'",
            "--exclude '.github'",
            "--exclude '.venv'",
            "--exclude '.DS_Store'",
            "--exclude '**/results/*'",
            "--exclude '**/figures/*'",
            "--exclude '**/tmp'",
            "--exclude '**/__pycache__'",
            "--exclude '**/.pytest_cache'",
            f"{LOC_PATH} {REM_PATH}",
        ]
    )
    subprocess.run(command, shell=True)


def install() -> None:
    """Build the environment on the cluster."""
    print("install")
    command = textwrap.dedent(
        """\
        module load {}
        source {}/.bash_profile
        source {}/.bashrc
        source {}/bin/activate
        cd {}
        pip install --no-index --upgrade pip
        pip install --no-index {}
        pip install --no-index --no-deps -e .
    """.format(
            " ".join(MODULES),
            HOME_DIR,
            HOME_DIR,
            VENV_DIR,
            BASE_DIR,
            " ".join(REQUIRES),
        )
    )
    subprocess.run(command, shell=True)


def make(name: str | None = None) -> None:
    """Prepare a campaign for running."""
    TMPS_DIR.mkdir(parents=True, exist_ok=True)
    for sweep_name, sweep in pick(name).items():
        experiment = sweep["experiment"]
        base = sweep.get("base")
        if not isinstance(base, Mapping) or not base:
            raise SystemExit(f"sweep '{sweep_name}' has no 'base'.")
        if not (EXPS_DIR / experiment / "run.py").exists():
            raise SystemExit(f"no experiment '{experiment}' to run it with")

        campaign = list(configs(base, sweep))
        if not campaign:
            raise SystemExit(f"sweep '{sweep_name}' describes no run")

        sweep_dir = TMPS_DIR / sweep_name
        if sweep_dir.is_dir():
            shutil.rmtree(sweep_dir)
        generate.write(sweep_dir / "configs", experiment, campaign)

        job_path = sweep_dir / JOB_NAME
        job_path.write_text(slurm_job_stream(sweep_name, sweep, len(campaign)))
        subprocess.run(f"chmod u+x {job_path}", shell=True)

        tasks = min(int(sweep["tasks"]), len(campaign))
        share = -(-len(campaign) // tasks)
        print(
            f"make {sweep_name}: {len(campaign)} configurations, "
            f"{tasks} tasks of at most {share}, {sweep['walltime']} each"
        )


def run(name: str) -> None:
    """Run one campaign."""
    job_path = TMPS_DIR / name / JOB_NAME
    if not job_path.is_file():
        raise SystemExit(f"{job_path} does not exist -- run 'make' first")
    subprocess.run(f"sbatch {job_path}", shell=True)


def receive(name: str | None = None) -> None:
    """Bring the records home, for one sweep or for all of them."""
    print("receive")
    experiments = sorted(
        {sweep["experiment"] for sweep in pick(name).values()}
    )
    if not experiments:
        return

    rules = []
    for experiment in experiments:
        rules += [f"--include={experiment}"]
        rules += [f"--include={experiment}/results"]
        rules += [f"--include={experiment}/results/**"]
    rules.append("--exclude=*")

    source = f"{REM_PATH}/mcrdv/experiments/"
    target = pathlib.Path(LOC_PATH, "experiments").expanduser()
    target.mkdir(parents=True, exist_ok=True)

    done = subprocess.run(["rsync", "-amv", *rules, source, str(target)])
    if done.returncode:
        raise SystemExit(f"receive failed: rsync exited {done.returncode}")


def clean() -> None:
    """Clean the generated campaign files, but not their records."""
    print("clean")
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)


# ----- Command line interface ----- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd",
        choices=[
            "send",
            "install",
            "make",
            "run",
            "receive",
            "clean",
        ],
    )
    parser.add_argument(
        "-s", "--sweep", default=None, help="one sweep of sweeps.yaml"
    )
    args = parser.parse_args()

    if args.cmd == "send":
        send()
    elif args.cmd == "install":
        install()
    elif args.cmd == "make":
        make(args.sweep)
    elif args.cmd == "receive":
        receive(args.sweep)
    elif args.cmd == "clean":
        clean()
    elif args.cmd == "run":
        if args.sweep is None:
            raise SystemExit("run needs a sweep: ./script.py run -s <name>")
        run(args.sweep)

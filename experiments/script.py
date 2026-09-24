#!/usr/bin/env python3
"""Running the experiments on the cluster, from one place.

    ./script.py send                 # local  -> cluster
    ./script.py install              # build the environment there
    ./script.py make                 # write every campaign and its job file
    ./script.py run -s sensitivity   # submit one campaign as an array job
    ./script.py status -s sensitivity  # did every configuration produce a record?
    ./script.py receive              # cluster -> local
    ./script.py clean                # forget the generated campaigns

`send` and `receive` run on the laptop, the rest on the cluster. What each
campaign is, and what it costs to run, is `sweeps.yaml`; this file only turns
that into configuration files, a Slurm job, and the two rsyncs around them.

A sweep describes its runs entirely: its `base`, adjusted by its `set`, is the
configuration its axes vary from, and no `run.yaml` is read. Those stay what
they are -- one run, by hand -- so that editing one cannot quietly change what
a campaign means.

A task runs its share of a campaign sequentially in one process, which is what
`run.py --task-id/--num-tasks` is for: numba compiles the solver in eight to
twelve seconds and then never again, so one configuration per array task would
pay that toll on every one of them. Configurations are spread across tasks by
stride rather than in blocks, since runs differ in cost by orders of magnitude
and a stride keeps the tasks balanced.
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
        raise SystemExit(
            f"unknown sweep '{name}' (known: {', '.join(sorted(known))})"
        )
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
    """Every configuration a sweep describes, arm by arm.

    A sweep's own `set` is applied to the base first, once: it is what lets a
    base be shared between sweeps through a YAML anchor and still be adjusted
    here, since an anchor can be referred to but not edited.

    `set` is also the one reserved axis name: there, each of its values is a
    bundle of overrides applied together, so quantities that must move as one
    do. As an axis it varies slowest, being the outermost loop of its arm.
    """
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


# ----- Slurm job file ----- #


def slurm_job_stream(name: str, sweep: Mapping[str, Any], count: int) -> str:
    """The array job that runs one campaign.

    The numba cache goes to node-local storage: it is written on every import
    and read on every task, which is the access pattern a shared filesystem
    handles worst.
    """
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
            raise SystemExit(
                f"sweep '{sweep_name}' has no 'base' configuration; a sweep "
                "describes its runs entirely, and reads no run.yaml"
            )
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


def _key(config: Mapping[str, Any]) -> str:
    """A configuration as one comparable string.

    A record keeps the configuration it ran verbatim, so a campaign's files and
    its records can be matched on their content rather than on a name, a count
    or a timestamp -- none of which survive a task dying halfway and the
    campaign being run again.
    """
    return json.dumps(config, sort_keys=True, default=str)


def status(name: str | None = None) -> None:
    """Say whether every configuration of a campaign produced a record.

    An array job that loses a task leaves a results directory that looks
    perfectly healthy: the records that are there are complete, and nothing
    says which ones are not. This is the check for that, and the reason to run
    it before `clean` -- afterwards the campaign's configurations are gone, and
    only the records themselves say what was run.
    """
    for sweep_name, sweep in pick(name).items():
        configs_dir = TMPS_DIR / sweep_name / "configs"
        if not configs_dir.is_dir():
            print(f"{sweep_name}: not made")
            continue

        wanted = {}
        for path in sorted(configs_dir.glob("run-*.yaml")):
            wanted[_key(configuration.load(path))] = path.name

        results = EXPS_DIR / sweep["experiment"] / "results"
        found, failed = set(), 0
        for item in recording.load(results) if results.is_dir() else ():
            found.add(_key(item.config))
            failed += bool(item.failed)

        missing = [name for key, name in wanted.items() if key not in found]
        done = len(wanted) - len(missing)
        state = "complete" if not missing else f"{len(missing)} MISSING"
        print(
            f"{sweep_name}: {done}/{len(wanted)} configurations have a record"
            f" -- {state}" + (f", {failed} failed" if failed else "")
        )
        for item in missing[:10]:
            print(f"    missing {item}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


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
            "status",
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
    elif args.cmd == "status":
        status(args.sweep)
    elif args.cmd == "receive":
        receive(args.sweep)
    elif args.cmd == "clean":
        clean()
    elif args.cmd == "run":
        if args.sweep is None:
            raise SystemExit("run needs a sweep: ./script.py run -s <name>")
        run(args.sweep)

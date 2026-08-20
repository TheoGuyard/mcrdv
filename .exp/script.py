import argparse
import json
import pathlib
import shutil
import subprocess
import textwrap


# ----- Global variables ----- #

REP_NAME = "scorpion"
LOC_PATH = "~/Documents/Github/{}".format(REP_NAME)
REM_PATH = "tguyard@rorqual.alliancecan.ca:links/scratch"
HOME_DIR = "/home/tguyard"

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
EXPS_DIR = BASE_DIR / "exp"
LOGS_DIR = EXPS_DIR / "logs"
TMPS_DIR = EXPS_DIR / "tmps"
RUN_PATH = EXPS_DIR / "run.sh"
JOB_FILE = "job.sh"

MODULES = ["rust"]


# ----- Job definitions ----- #

DATASETS = {
    "cosmos1408.csv": {
        "num_debris"    : 4,
        "max_time"      : 365 * 24 * 3_600.0,
        "max_fuel"      : 25_000.0,
        "max_load"      : 10,
        "chaser_init"   : "centroid",
        "chaser_load"   : 1.5,
    },
    "odrc.csv": {
        "num_debris"    : 13,
        "max_time"      : 365 * 24 * 3_600.0,
        "max_fuel"      : 25_000.0,
        "max_load"      : 10,
        "chaser_init"   : "centroid",
        "chaser_load"   : 1.5,
    },
    "iridium33.csv": {
        "num_debris"    : 110,
        "max_time"      : 365 * 24 * 3_600.0,
        "max_fuel"      : 25_000.0,
        "max_load"      : 10,
        "chaser_init"   : "centroid",
        "chaser_load"   : 1.5,
    },
    "cosmos2251.csv": {
        "num_debris"    : 583,
        "max_time"      : 365 * 24 * 3_600.0,
        "max_fuel"      : 25_000.0,
        "max_load"      : 10,
        "chaser_init"   : "centroid",
        "chaser_load"   : 1.5,
    },
    "fengyun1C.csv": {
        "num_debris"    : 1_847,
        "max_time"      : 365 * 24 * 3_600.0,
        "max_fuel"      : 25_000.0,
        "max_load"      : 10,
        "chaser_init"   : "centroid",
        "chaser_load"   : 1.5,
    }
}

JOB_ARRAYS = [

    # -------------------- Test experiment -------------------- #
    
    {
        "name"      : "simple",
        "walltime"  : "00:05:00",
        "memory"    : f"{1 * 1024:d}M",
        "cpus"      : 1,
        "save"      : False,
        "configs"   : [
            {
                "experiment": {
                    "name"          : "simple",
                },
                "problem"   : {
                    "path"          : "data/iridium33.csv",
                    "max_time"      : 365 * 24 * 3600.0,
                    "max_fuel"      : 25_000.0,
                    "max_load"      : 10,
                    "factor_time"   : 0.0,
                    "factor_fuel"   : 1.0,
                    "chaser_init"   : "centroid",
                    "chaser_load"   : 1.5
                },
                "solver"    : {
                    "type"          : "scorpion",
                    "args"          : {
                        "inner" : {
                            "type"  : "qlaw",
                            "args"  : {
                                "max_thrust"        : 0.003
                            }
                        },
                        "middle": {
                            "type"  : "dynprog",
                            "args"  : {
                                "allow_waiting"     : True,
                                "grid_steps"        : 12,
                                "departure_hints"   : True
                            }
                        },
                        "outer" : {
                            "type"  : "hgs",
                            "args"  : {
                                "seed"              : 42,
                                "preset"            : 6,
                                "time_limit"        : 6 * 3_600.0,
                                "generation"        : "greedy",
                            }
                        }
                    }
                }
            }
        ],
    },

    # -------------------- Analysis experiment -------------------- #
    
    {
        "name"      : "analysis",
        "walltime"  : "06:30:00",
        "memory"    : f"{8 * 1024:d}M",
        "cpus"      : 8,
        "save"      : True,
        "configs"   : [
            {
                "experiment": {
                    "name"          : "analysis",
                },
                "problem"   : {
                    "path"          : "data/" + dataset_name,
                    "max_time"      : dataset_params["max_time"],
                    "max_fuel"      : dataset_params["max_fuel"],
                    "max_load"      : dataset_params["max_load"],
                    "factor_time"   : factor_time,
                    "factor_fuel"   : factor_fuel,
                    "chaser_init"   : dataset_params["chaser_init"],
                    "chaser_load"   : dataset_params["chaser_load"]
                },
                "solver"    : {
                    "type"          : "scorpion",
                    "args"          : {
                        "inner" : {
                            "type"  : "qlaw",
                            "args"  : {
                                "max_thrust"        : 0.003
                            }
                        },
                        "middle": {
                            "type"  : "dynprog",
                            "args"  : {
                                "allow_waiting"     : True,
                                "grid_steps"        : 12,
                                "departure_hints"   : True,
                            }
                        },
                        "outer" : {
                            "type"  : "hgs",
                            "args"  : {
                                "seed"              : 42,
                                "preset"            : 6,
                                "time_limit"        : 6 * 3_600.0,
                                "generation"        : "greedy",
                            }
                        }
                    }
                }
            }
            for dataset_name, dataset_params in DATASETS.items()
            for (factor_time, factor_fuel) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        ],
    },
]


# ----- Slurm bash files ----- #


def slurm_run_stream():
    steam = textwrap.dedent("""\
        #!/bin/sh
        expname=$1
        repeats=$2
        for i in $(seq 1 $repeats);
        do
            sbatch {}/$expname/{}
        done
    """.format(TMPS_DIR, JOB_FILE))
    return steam

def slurm_job_stream(job_array: dict):
    stream = textwrap.dedent("""\
        #!/bin/sh
        #SBATCH -J {}_{}
        #SBATCH -o {}/%x_%A_%a.out    
        #SBATCH -e {}/%x_%A_%a.err
        {}
        {}
        {}
        #SBATCH --array=0-{}    
        #SBATCH --account=def-vidalthi

        config_file="{}_${{SLURM_ARRAY_TASK_ID}}.json"
        
        module load {}
        source {}/.bash_profile

        cd {}
        cargo run --release --example exp -- {}/{}/configs/$config_file {}
    """.format(
        REP_NAME,
        job_array["name"],
        LOGS_DIR,
        LOGS_DIR,
        "#SBATCH -t {}".format(job_array["walltime"]) if "walltime" in job_array else "",
        "#SBATCH --mem-per-cpu={}".format(job_array["memory"]) if "memory" in job_array else "",
        "#SBATCH --cpus-per-task={}".format(job_array["cpus"]) if "cpus" in job_array else "",
        len(job_array["configs"]) - 1,
        job_array["name"],
        " ".join(MODULES),
        HOME_DIR,
        BASE_DIR,
        TMPS_DIR,
        job_array["name"],
        "--save" if job_array.get("save", True) else "",
    ))
    return stream


# ----- Scripts functions ----- #

def send():
    print("send")
    cmd_str = " ".join(
        [
            "rsync -amv",
            "--exclude '.DS_Store'",
            "--exclude '.git'",
            "--exclude 'target'",
            "--exclude '**/results/*.json'",
            "{} {}".format(LOC_PATH, REM_PATH),
        ]
    )
    subprocess.run(cmd_str, shell=True)

def receive():
    print("receive")
    src_path = pathlib.Path(REM_PATH, REP_NAME, "exp", "results/*")
    dst_path = pathlib.Path(LOC_PATH, "exp", "results")
    cmd_str = "rsync -amv {} {}".format(src_path, dst_path)
    subprocess.run(cmd_str, shell=True)

def make():

    clean()

    # Top-level run file
    with open(RUN_PATH, "w") as file:
        file.write(slurm_run_stream())
    subprocess.run("chmod u+x {}".format(RUN_PATH), shell=True)

    # Job-level config and launch file in temporary dir
    TMPS_DIR.mkdir()
    for job_array in JOB_ARRAYS:
        print("make {}".format(job_array["name"]))

        # Create job dir
        job_array_dir = TMPS_DIR.joinpath(job_array["name"])
        job_array_dir.mkdir()

        # Create job config files
        job_array_configs_dir = job_array_dir.joinpath("configs")
        job_array_configs_dir.mkdir()
        for i, config in enumerate(job_array["configs"]):
            config_file = f"{job_array['name']}_{i}.json"
            config_path = job_array_configs_dir.joinpath(config_file)
            with open(config_path, "w") as file:
                json.dump(config, file)

        # Create job launch file
        job_array_path = job_array_dir.joinpath(JOB_FILE)
        with open(job_array_path, "w") as file:
            file.write(slurm_job_stream(job_array))
        subprocess.run("chmod u+x {}".format(job_array_path), shell=True)

    # Add logs dir if not existing
    if not LOGS_DIR.is_dir():
        LOGS_DIR.mkdir()

def clean():
    print("clean")
    if RUN_PATH.is_file():
        RUN_PATH.unlink()
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)

def run(expname, repeats=1):
    subprocess.run("{} {} {}".format(RUN_PATH, expname, repeats), shell=True)


# ----- Command line interface ----- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["send", "receive", "make", "clean", "run"])
    parser.add_argument("-e", "--expname", default=None)
    parser.add_argument("-r", "--repeats", type=int, default=1)
    args = parser.parse_args()

    if args.cmd == "send":
        send()
    elif args.cmd == "receive":
        receive()
    elif args.cmd == "make":
        make()
    elif args.cmd == "clean":
        clean()
    elif args.cmd == "run":
        if args.expname is None:
            raise ValueError("Experiment name must be provided for receive.")
        run(args.expname, args.repeats)
    else:
        raise ValueError(f"Unknown command {args.cmd}.")
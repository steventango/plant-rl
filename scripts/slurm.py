import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")
import argparse
import dataclasses
import math
import time
from functools import partial

import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.runner.utils import approximate_cost, gather_missing_indices
from PyExpUtils.utils.generator import group
from runner.Slurm import (
    SingleNodeOptions,
    buildParallel,
    fromFile,
    get_script_name,
    schedule,
    to_cmdline_flags,
)

import experiment.ExperimentModel as Experiment
#from utils.results import gather_missing_indices

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", type=str, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--entry", type=str, default="src/main.py")
parser.add_argument("--results", type=str, default="./")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--force", action="store_true", default=False)
parser.add_argument("--exclude", type=str, nargs="+", default=[])
parser.add_argument("-i", "--idxs", type=int, nargs="+", default=None)
parser.add_argument("--time", type=str, default=None)

cmdline = parser.parse_args()

ANNUAL_ALLOCATION = 724

# -------------------------------
# Generate scheduling bash script
# -------------------------------
cwd = os.getcwd()
project_name = os.path.basename(cwd)

venv_origin = ".venv"
venv = "$SLURM_TMPDIR"


# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
def getJobScript(parallel: str, slurm):
    if len(cmdline.exclude) > 0:
        exclude_str = "#SBATCH --exclude=" + ",".join(cmdline.exclude)
    else:
        exclude_str = ""
    if slurm.gpus:
        device_str = """export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d"""
    else:
        device_str = "export JAX_PLATFORMS=cpu"
    jobs = math.ceil(int(slurm.cores / slurm.threads_per_task) * slurm.tasks_per_core / slurm.tasks_per_vmap)
    max_xla_python_client_mem_fraction = 0.95 if jobs == 1 else 0.3
    return f"""#!/bin/bash

{exclude_str}
cd {cwd}
cp -R {venv_origin} {venv}

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION={max_xla_python_client_mem_fraction / jobs}
{device_str}

{parallel}
    """


# -----------------
# Environment check
# -----------------
if not cmdline.debug and not os.path.exists(venv_origin):
    print("WARNING: zipped virtual environment not found at:", venv_origin)
    print("Make sure to run `scripts/setup_cc.sh` first.")
    exit(1)

# ----------------
# Scheduling logic
# ----------------
slurm = fromFile(cmdline.cluster)

# Override time if provided
if cmdline.time is not None:
    slurm = dataclasses.replace(slurm, time=cmdline.time)

threads = slurm.threads_per_task if isinstance(slurm, SingleNodeOptions) else 1
tasks_per_core = slurm.tasks_per_core if isinstance(slurm, SingleNodeOptions) else 1

# compute how many "tasks" to clump into each job
base_group_size = math.ceil(slurm.cores / threads * tasks_per_core) * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(":")
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing or use provided indices
if cmdline.idxs is not None:
    missing = {path: cmdline.idxs for path in cmdline.e}
else:
    missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load)

# compute cost
memory = Slurm.memory_in_mb(slurm.mem_per_core)
compute_cost = partial(
    approximate_cost, cores_per_job=slurm.cores, mem_per_core=memory, hours=total_hours
)
cost = sum(
    compute_cost(math.ceil(len(job_list) / base_group_size)) for job_list in missing.values()
)
perc = (cost / ANNUAL_ALLOCATION) * 100

print(
    f"Expected to use {cost:.2f} core years, which is {perc:.4f}% of our annual allocation"
)
if not cmdline.debug and not cmdline.force:
    input("Press Enter to confirm or ctrl+c to exit")

# start scheduling
for path in missing:
    total_tasks = len(missing[path])
    if total_tasks == 0:
        continue

    # compute how many "tasks" to clump into each job
    num_jobs = math.ceil(total_tasks / base_group_size)
    groupSize = math.ceil(total_tasks / num_jobs)

    for g in group(missing[path], groupSize):
        job_indices = list(g)
        print("scheduling:", path, job_indices)
        # make sure to only request the number of CPU cores necessary
        tasks = min([groupSize, len(job_indices)])
        par_tasks = max(math.ceil(tasks / slurm.sequential), 1)
        cores = math.ceil(par_tasks * threads / tasks_per_core)
        sub = dataclasses.replace(slurm, cores=cores)

        # build the executable string
        # instead of activating the venv every time, just use its python directly
        gpu_str = "--gpu" if sub.gpus else ""
        runner = f"{venv}/.venv/bin/python {cmdline.entry} {gpu_str} -e {path} --save_path {cmdline.results} --checkpoint_path=$SCRATCH/checkpoints/{project_name} -i "

        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = buildParallel(runner, job_indices, sub)

        # generate the bash script which will be scheduled
        script = getJobScript(parallel, sub)
        script_name = get_script_name(Path(path), job_indices)
        print(script_name)

        if cmdline.debug:
            print(to_cmdline_flags(sub))
            print(script)
            exit()

        schedule(script, sub, script_name)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
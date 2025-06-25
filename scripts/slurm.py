import os
import sys

sys.path.append(os.getcwd() + "/src")

import argparse
import dataclasses
import math
from functools import partial

import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.runner.utils import approximate_cost, gather_missing_indices
from PyExpUtils.utils.generator import group

import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", type=str, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--entry", type=str, default="src/main.py")
parser.add_argument("--results", type=str, default="./")
parser.add_argument("--debug", action="store_true", default=False)

cmdline = parser.parse_args()

# -------------------------------
# Generate scheduling bash script
# -------------------------------


# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
def getJobScript(parallel):
    return f"""#!/bin/bash

#SBATCH --signal=B:SIGTERM@180

module load apptainer

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1

{parallel}
    """


# ----------------
# Scheduling logic
# ----------------
slurm = Slurm.fromFile(cmdline.cluster)

threads = slurm.threads_per_task if isinstance(slurm, Slurm.SingleNodeOptions) else 1

# compute how many "tasks" to clump into each job
groupSize = int(slurm.cores / threads) * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(":")
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing
missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load)

# compute cost
memory = Slurm.memory_in_mb(slurm.mem_per_core)
compute_cost = partial(
    approximate_cost, cores_per_job=slurm.cores, mem_per_core=memory, hours=total_hours
)
cost = sum(
    compute_cost(math.ceil(len(job_list) / groupSize)) for job_list in missing.values()
)

print(f"Expected to use {cost:.2f} core years.")
if not cmdline.debug:
    input("Press Enter to confirm or ctrl+c to exit")

# create directory to save all the scripts, if it doesn't exist
os.makedirs("slurm_scripts", exist_ok=True)

# generate submission script
submit_all = """#!/bin/bash
"""

# start scheduling
for path in missing:
    for g in group(missing[path], groupSize):
        l = list(g)
        print("scheduling:", path, f"{min(l)}-{max(l)}")
        # make sure to only request the number of CPU cores necessary
        tasks = min([groupSize, len(l)])
        par_tasks = max(int(tasks // slurm.sequential), 1)
        cores = par_tasks * threads
        sub = dataclasses.replace(slurm, cores=cores)

        # build the executable string
        # instead of activating the venv every time, just use its python directly
        runner = f"apptainer exec -C -B .:${{HOME}} -W ${{SLURM_TMPDIR}} pyproject.sif python {cmdline.entry} -e {path} --save_path {cmdline.results} -i "

        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Slurm.buildParallel(runner, l, sub)

        # generate the bash script
        script = getJobScript(parallel)

        if cmdline.debug:
            print(f"sub={Slurm.to_cmdline_flags(sub)}")
            print(f"script={script}")
            exit()

        script_name = f"slurm_scripts/job_{min(l)}-{max(l)}.sh"
        with open(script_name, "w") as f:
            f.write(script)
        os.chmod(script_name, 0o755)

        submit_all += f"sbatch {Slurm.to_cmdline_flags(sub)} {script_name}\n"
        submit_all += "sleep 2\n"

with open("slurm_scripts/submit_all.sh", "w") as f:
    f.write(submit_all)
os.chmod("slurm_scripts/submit_all.sh", 0o755)

print("\nTo submit all jobs, run:")
print("./slurm_scripts/submit_all.sh")

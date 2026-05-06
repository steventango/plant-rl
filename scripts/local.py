import os
import sys
from math import ceil

sys.path.append(os.getcwd() + "/src")
import argparse
import random
import subprocess
from functools import partial
from multiprocessing.pool import Pool

import experiment.ExperimentModel as Experiment
from utils.results import gather_missing_indices

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--entry", type=str, default="src/main.py")
parser.add_argument("--results", type=str, default="./")
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--vmap", type=int, default=1)


def count(pre, it):
    print(pre, 0, end="\r")
    for i, x in enumerate(it):
        print(pre, i + 1, end="\r")
        yield x

    print()


if __name__ == "__main__":
    cmdline = parser.parse_args()

    max_processes = 1
    pool = Pool(max_processes)

    env = os.environ.copy()

    cmds = []
    e_to_missing = gather_missing_indices(
        cmdline.e, cmdline.runs, loader=Experiment.load
    )
    for path in cmdline.e:
        exp = Experiment.load(path)

        indices = list(count(path, e_to_missing[path]))
        n = len(indices)
        if not n:
            continue
        processes = min(ceil(n / cmdline.vmap), max_processes)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(0.9 / processes)

        if cmdline.gpu:
            exe = f"python {cmdline.entry} --gpu --silent -e {path} -i "
            env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
            env["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
            subprocess.run("nvidia-cuda-mps-control -d", shell=True)
        else:
            env["JAX_PLATFORM_NAME"] = "cpu"
            env["JAX_PLATFORMS"] = "cpu"
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["NPROC"] = "1"
            env["XLA_FLAGS"] = (
                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
            )
            exe = f"python {cmdline.entry} --silent -e {path} -i "

        def batched(arr, batch_size: int):
            for i in range(0, len(arr), batch_size):
                yield arr[i : i + batch_size]

        for group in batched(indices, cmdline.vmap):
            idxs = " ".join([str(idx) for idx in group])
            cmds.append(exe + idxs)

    print(len(cmds))
    random.shuffle(cmds)
    res = pool.imap_unordered(
        partial(subprocess.run, shell=True, stdout=subprocess.PIPE, env=env),
        cmds,
        chunksize=1,
    )
    for i, _ in enumerate(res):
        sys.stderr.write(f"\r{i + 1}/{len(cmds)}")
    sys.stderr.write("\n")

    pool.close()

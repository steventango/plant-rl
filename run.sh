export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads

python scripts/local.py --runs 1 -e experiments/offline/S8/P3/InAC_LN.json --entry src/main_offline.py --gpu
python scripts/local.py --runs 1 -e experiments/offline/S8/P3/Constant_White.json --entry src/main_offline.py --gpu
python scripts/local.py --runs 1 -e experiments/offline/S8/P3/Constant_Red.json --entry src/main_offline.py --gpu
python scripts/local.py --runs 1 -e experiments/offline/S8/P3/Constant_Blue.json --entry src/main_offline.py --gpu
python scripts/local.py --runs 1 -e experiments/offline/S8/P3/Dirichlet_Uniform.json --entry src/main_offline.py --gpu

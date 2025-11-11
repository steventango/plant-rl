export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.25.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.5.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.75.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim1.json -i 0 &
wait

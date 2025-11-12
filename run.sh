export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads
python src/main_offline.py -e experiments/offline/S8/P0/InAC.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.25.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.5.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.75.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim1.json -i 0 &
wait
python src/main.py -e experiments/offline/S8/P0/InAC_eval.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0_eval.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.25_eval.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.5_eval.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim0.75_eval.json -i 0 &
python src/main.py -e experiments/offline/S8/P0/InAC_GPsim1_eval.json -i 0 &
wait
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC/0 &
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC_GPsim0/0 &
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC_GPsim0.25/0 &
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC_GPsim0.5/0 &
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC_GPsim0.75/0 &
python src/visualize_simplex_policy.py --exp_path results/offline/S8/P0/InAC_GPsim1/0 &
wait

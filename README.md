# Optimal Greenhouse Lighting by Reinforcement Learning

This repository is based on andnp/rl-control-template. Please see its README.md for a (slightly outdated) installation and user guide.

## Usage
Run a random agent in PlantGrowthChamber

```bash
python src/main_real.py -e experiments/online/E1/P0/Random.json -i 0
```


To deploy in all zones
```bash
docker compose up -d zone1 zone2 zone3 zone4 zone5 zone6 zone7 zone8 zone9 zone10 zone11 zone12
```

## Schedule Slurm jobs
To schedule an experiment on an Alliance Canada cluster, use slurm.py:
1. Go to your project directory. It needs to have the apptainer image `plantRL.sif`
2. Set your compute requirements in `clusters/compute.json` (or make another file)
3. Run the following (edit it for your job)
```
module load apptainer
apptainer exec -C -B .:$HOME plantRL.sif python scripts/slurm.py --cluster clusters/compute.json --runs 5 -e experiments/offline/E1/P2/GAC.json
```

4. The slurm scripts will be saved in `slurm_scripts/`. To submit them all, run `./slurm_scripts/submit_all.sh`.

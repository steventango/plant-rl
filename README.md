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
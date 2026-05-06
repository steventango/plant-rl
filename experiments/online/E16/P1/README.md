# Experiment E16 / Phase P1

Test whether there is a statistical difference between the returns of the different policies. Verify that there is no differences between the returns of the constant policy deployed in different zones.

## Environment
### PlantGrowthChamber
#### State
- Plant solidity
- Log clean area
- Liters per pot
- DINOv3 embedding PC 1
- DINOv3 embedding PC 2
- DINOv3 embedding PC 3
- DINOv3 embedding PC 4

#### Actions
- Dirichlet policy over the Color Triangle: (Red, White, Blue)

#### Time step
1 day

## Agents
### Constant
- **Zones 1, 5, 9**
- Running with `Constant1`, `Constant5`, and `Constant9` configurations.

### InAC
- **Zones 2, 3, 4, 6, 7, 8, 10, 11, 12**
- Pretrained models from `results/offline/S8/P2/InAC_LN/`.
- Testing three different policies learned with offline RL.

## Deployment
- Using alliance chambers (zones 1-12).
- Start Date: 2026-02-04 09:30 AM (Europe/Athens).

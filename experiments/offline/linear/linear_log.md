# Experiments with linear function approximation in the MultiPlantSimulator env
## <u>Linear1</u>
Debugging Q-learning and ESARSA, Q-learning seems to perform much better

#### Notes
- One hot time does not work as well as sine time

#### TODO
- Implemented Q-learning without subclassing LinearAgent, for debugging 
purposes since ESARSA wasn't working (the issue was bad initialization values)
probably want to change this to subclass LinearAgent in the future. 

## <u>Linear2</u>
Full Q-learning sweep

#### Notes
- Best parameters are: ___
- Annealing epsilon seems to be important 

#### TODO
Get plots with all values of annealing

## <u>Linear3</u>
Repeat of Linear2 with 32 plants to demonstrate that using the full 49 is better 
(reduces noise in sensor readings by taking mean over all plants). This is fine 
becuase we are no longer splitting the chamber anyways (we will use all 64 plants for each agent)

#### Notes
- Best parameters are: ___

#### TODO

## <u>Linear4</u> (NOT DONE YET, ignore the configs in here for now)
DQN agent with single linear layer (to test batch updating for greater sample efficiency, though a confounding factor here 
is Andy's DQN uses ADAM not regular SGD, might want to change to isolate the effect of batch updates but probably not essential). Use larger batch size, maybe increase number of updates per step? 

#### Notes
- Best parameters are: ___

#### TODO

## <u>Linear5</u> (NOT DONE YET, dir doesn't exist)
GAC with a single linear layer and large batch sizes, maybe increase number of updates per step? 

#### Notes
- Best parameters are: ___

#### TODO
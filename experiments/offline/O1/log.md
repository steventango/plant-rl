# Experiments in the PlantSimulator Env (Oliver)
## <u>E0-Linear1</u>
Debugging Q-learning and ESARSA, Q-learning seems to perform much better

#### Results
- One hot time does not work as well as sine time

#### TODO
- Implemented Q-learning without subclassing LinearAgent, for debugging 
purposes since ESARSA wasn't working (the issue was bad initialization values)
probably want to change this to subclass LinearAgent in the future. 



## <u>E1-Linear2</u>
Q-learning sweep (linear function approx). Learning from 1 day of experience (72 steps). Stride and lag set to 1. 

#### Results
- Best parameters are: ___

#### TODO
None



## <u>E2-DQN</u>

DQN agent with single linear layer (to test batch updating for greater sample efficiency, though a confounding factor here is Andy's DQN uses ADAM not regular SGD, might want to change to isolate the effect of batch updates but probably not essential).

Testing large batch sizes. Target refresh rate is set to 1 since this is just q-learning with linear function approx updating in batch with Adam. Should be ok?

Made new DQN class that allows for specifying a minimum batch size, and still doing updates as long as the number of collected samples is greater than the minimum batch size (even if it is less than the full batch size, which could be quite large)

#### Results
- Best parameters are: ___

#### TODO
Test multiple updates per step?


## <u>E3-DQN</u>
DQN with hidden layer size 4. Large batch sizes like in E2 above.

#### Results
- Best parameters are: ___

#### TODO
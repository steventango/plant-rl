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


## <u>E2-linearDQN</u>

DQN agent with single linear layer (to test batch updating for greater sample efficiency, though a confounding factor here is Andy's DQN uses ADAM not regular SGD, might want to change to isolate the effect of batch updates but probably not essential).

Testing large batch sizes. Target refresh rate is set to 1 since this is just q-learning with linear function approx updating in batch with Adam. Should be ok?

Made new DQN class that allows for specifying a minimum batch size, and still doing updates as long as the number of collected samples is greater than the minimum batch size (even if it is less than the full batch size, which could be quite large)

#### Results
- Best hypers:
target_refresh:     1
epsilon:            0.1
n_step:             1
buffer_size:        2000
optimizer.beta2:    0.999
min_batch:          32
optimizer.name:     ADAM
buffer_type:        uniform
optimizer.beta1:    0.9
optimizer.alpha:    0.01
batch:              128

Failed to learn in a week. Selecting action 0 much more than 10% of the time. 

#### TODO
Test multiple updates per step? Going to test on 1 day period to see if we get similar results to ESARSA in that learning over longer timescale is harder. Also want to try with regular sgd (no adam)


## <u>E3-DQN</u>
DQN with hidden layer size 4. One week experiment. Large batch sizes like in E2 above.

#### Results
- Best hypers:
buffer_type:              uniform
batch:                    512
optimizer.beta1:          0.9
buffer_size:              2000
n_step:                   1
min_batch:                32
epsilon:                  0.1
representation.type:      OneLayerRelu
optimizer.name:           ADAM
representation.hidden:    4
optimizer.beta2:          0.999
optimizer.alpha:          0.1
target_refresh:           1

Failed to learn in a week. Did much worse than linear. Notably I had the target refresh rate set to one accidentally which was probably bad even with the single hidden layer with 4 units, not sure. There's only like 500ish datapoints so not sure what is a good value. 

#### TODO
Going to sweep a more conventional range for batch size. Also will try FTA and different target refresh rates.

## <u>E4-linearDQN</u>

DQN agent with single linear layer. Testing on 1 day period with ADAM and SGD (technically adam with beta1=0 and beta2=1)

#### Results
- Best hypers:
n_step:             1
buffer_type:        uniform
optimizer.beta1:    0.9
target_refresh:     1
min_batch:          8
batch:              8
buffer_size:        2000
optimizer.beta2:    0.999
epsilon:            0.1
optimizer.name:     ADAM
optimizer.alpha:    0.1

Both failed to learn in a day. ADAM with momentum did better. 
Notably the smallest batch size did the best, interesting because 128 did best on one week timescale. Makes me think it's a matter of what transitions you're working with since some are noisier than others. 

#### TODO
Multiple updates per step. Maybe use 2 day period instead (Steven asked for this for experiment this weekend). Be sure to include small batch size too.


## <u>E4-linearDQN</u>
Comparing SGD to ADAM for linear DQN

#### Results
ADAM a bit better

#### TODO
None

## <u>E4-DQN_fta</u>
Implemented DQN with FTA. 

#### Results
Didn't do well on 12 hr problem. 

#### TODO
None

## <u>E6-DQN_sweeps</u>
Big DQN sweep over batch size and learning rate with hypers decided by ADAM (the prof not the optimizer). Used 16k steps. 

#### Results
Very bad. 

#### TODO
None

## <u>E7-trivial_rew_1ep</u>
Testing ESARSA with tile coding and DQN in the trivial reward env. The reward is just +1 if selected action matches the twilight policy else -1. 

Note that DQN uses sin/cos time encoding while ESARSA uses linear time since it is being tile coded. 

DQN just uses the hypers we discussed in meeting. 

Total exp len is just two weeks (one episode). 


#### Results
ESARSA solves the task easily. 

#### TODO






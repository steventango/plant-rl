import pytest
from environments.PlantSimulator import PlantSimulator
import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save
import os

@pytest.fixture
def env():
    return PlantSimulator()

def test_plant_data_analysis(env): 
    observed_area = np.reshape(env.data, (-1, env.steps_per_day))  # reshape into different days
    max_indices = np.argmax(observed_area, axis=1)        # index at the max value of each day
    
    actual_area = np.copy(env.data)
    for i in range(observed_area.shape[0]-1):
        min_id = i*env.steps_per_day + max_indices[i]
        max_id = (i+1)*env.steps_per_day + max_indices[i+1]
        actual_area[min_id:max_id] = np.interp(np.arange(min_id, max_id), [min_id, max_id], [env.data[min_id],env.data[max_id]])
    
    projection_factor = np.reshape(env.data/actual_area, (-1, env.steps_per_day))
    for i in range(observed_area.shape[0]):
        row = projection_factor[i]
        plt.plot(row, alpha=1/16*i, color='k', linewidth=1)
    plt.plot(np.mean(projection_factor, axis=0), 'r', label='average projection factor')
    plt.title('Leaf area projection factor (light to dark -> day 1 to 16)')
    plt.xlabel(f'Time of Day ({env.steps_per_day} intervals)')
    plt.ylabel(f'Observed/Actual Leaf Area')
    plt.savefig('tests/plots/plant_simulator_projection_factor.png')
    plt.show()
    
def test_light_random_policy(env, off_prob=0):
    env.start()
    R = []
    S = []
    A = []
    for _ in range(len(env.actual_area) - 1):
        reward, next_state, done, info = env.step(np.random.choice([0, 1], p = [off_prob, 1-off_prob]))
        A.append(env.actual_area[env.num_steps])
        R.append(reward)
        S.append(next_state[1])
    
    plt.plot(R, label=f'return={np.sum(R):.2f}')
    plt.title(f'Reward history if agent turns off light with probability={off_prob}')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('tests/plots/plant_simulator_reward_history.png')
    plt.show()
    plt.plot(S, 'b', label='observed leaf area')
    plt.plot(A, 'r', label=f'actual leaf area, total gain={A[-1]-A[0]:.2f}')
    plt.title(f'Leaf area history if agent turns off light with probability={off_prob}')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Leaf Area')
    plt.legend()
    plt.savefig('tests/plots/plant_simulator_area_history.png')
    plt.show()






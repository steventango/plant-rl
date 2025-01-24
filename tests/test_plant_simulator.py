import pytest
from environments.PlantSimulator import PlantSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@pytest.fixture
def env():
    return PlantSimulator()

def return_area_correlation(env):
    ''' See if return and total gain in area are positively correlated '''
    episodic_return = []
    area_gain = []
    for off_prob in np.linspace(0, 1, 40):
        env.start()
        R = []; S = []; A = []
        for _ in range(len(env.actual_area) - 1):
            reward, next_state, done, info = env.step(np.random.choice([0, 1], p = [off_prob, 1-off_prob]))
            A.append(env.actual_area[env.num_steps])
            R.append(reward)
        episodic_return.append(np.sum(R))
        area_gain.append(A[-1]-A[0])
    
    plt.plot(np.linspace(0, 1, 40), episodic_return, label=f'return')
    plt.plot(np.linspace(0, 1, 40), area_gain, label=f'total area gain')
    plt.title(f'Return and Total Area Gain for Various Random Policies')
    plt.xlabel('Light Off Probability')
    plt.ylabel('Return or Area Gain')
    plt.legend()
    plt.savefig('tests/plots/test_plant_simulator/return_area_correlation.png')
    plt.show()

def test_random_light_policy(env, off_prob=0.5):
    ''' Test out PlantSimulator environment with random policy '''
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
    plt.savefig('tests/plots/test_plant_simulator/reward_history_light-random.png')
    plt.show()
    plt.plot(S, 'b', label='observed leaf area')
    plt.plot(A, 'r', label=f'actual leaf area, total gain={A[-1]-A[0]:.2f}')
    plt.title(f'Leaf area history if agent turns off light with probability={off_prob}')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Leaf Area')
    plt.legend()
    plt.savefig('tests/plots/test_plant_simulator/area_history_light-random.png')
    plt.show()

def interpolate_actual_area(env):   
    ''' Linear interpolation between max observed areas of consecutive days '''

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
    plt.ylabel(f'Observed / Actual Leaf Area')
    plt.savefig('tests/plots/test_plant_simulator/projection_factor.png')
    plt.show()

def plant_data_analysis(plant_id = 1):
    ''' Playing with raw plant data '''

    df = pd.read_csv('src/environments/plant_all_data.csv')
    sub_df = df[df['plant_id'] == plant_id]  
    sub_df = sub_df.sort_values(by='timestamp')

    area = np.array(sub_df['area'])
    peri = np.array(sub_df['perimeter'])

    # Moving average
    window_size = 1
    area = np.convolve(area, np.ones(window_size)/window_size, mode='valid')
    peri = np.convolve(peri, np.ones(window_size)/window_size, mode='valid')
    ratio = area / peri

    # Normalize for comparison
    ratio = (ratio-np.min(ratio))/(np.max(ratio)-np.min(ratio))
    area = (area-np.min(area))/(np.max(area)-np.min(area))
    peri = (peri-np.min(peri))/(np.max(peri)-np.min(peri))

    #plt.plot(peri, label='peri', linewidth=0.5)
    plt.plot(area, label='area', linewidth=0.5)
    #plt.plot(ratio, label='ratio', linewidth=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Growth Indicator')
    plt.legend()
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = ['red', 'blue']
AGENTS = ['Constant1/z1', 'Constant2/z2']
NAMES = ['constant dim', 'constant bright']
TIME_STEP = [1, 3, 6, 12]
TIME_STEP_NAME = ['10min', '30min', '1hr', '2hr']

f, ax = plt.subplots(4, 1)
for t in range(4):
    time_step = TIME_STEP[t] 

    for i in [1,2]:
        agent = f'Constant{i}/z{i}'
        data_path = f'/data/online/E7/P2/Constant{i}/z{i}/raw.csv'
        full_df = pd.read_csv(data_path)
        unique_episodes = sorted(full_df['episode'].unique())

        R = []
        TIME = []
        for j in unique_episodes[1:-1]:
            df = full_df[full_df['episode'] == j]        

            areas = df['area'].to_numpy()
            areas = np.reshape(areas, (-1,36))
            area = np.array([np.mean(areas[i,:]) for i in range(areas.shape[0])])

            reward = area[time_step:] - area[:-time_step]
            R.append(reward)
            # Convert time strings to datetime and format as month:day:hour:minute
            time_series = pd.to_datetime(df['time']).unique()
            formatted_times = [t.strftime('%m-%d-%H-%M') for t in time_series[:-time_step]]
            TIME.append(formatted_times)

        R = np.hstack(R)
        TIME = np.hstack(TIME)
                
        ax[t].plot(TIME, R, color=COLORS[i-1], label=NAMES[i-1], linewidth=0.5)

        ax[t].set_ylabel('Reward')
        ax[t].set_title(f"{TIME_STEP_NAME[t]} Time Step")
        ax[t].set_xlabel('Time')
        if t == 3:  # Only show x-axis labels for the bottom subplot
            ax[t].set_xlabel('Time')

        ax[t].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if t == 0:  # Only add the legend to the top subplot
            ax[t].legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
f.tight_layout()
plt.savefig('/data/scripts/E7_rew_plots', dpi=300, bbox_inches='tight')
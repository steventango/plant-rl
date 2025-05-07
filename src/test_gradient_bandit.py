#!/usr/bin/env python3
"""
Test script for replicating Figure 2.5 from Sutton & Barto's Reinforcement Learning book.

This script tests the gradient bandit algorithm with and without a reward baseline
on the 10-armed testbed when the q*(a) are chosen to be near +4 rather than near zero.
"""

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from environments.KArmedBandit import KArmedBandit


def run_gradient_algorithm(bandits, alpha=0.1, use_baseline=True):
    """
    Run the gradient bandit algorithm on a set of bandit problems.

    Parameters:
    -----------
    bandits : list
        List of KArmedBandit environments
    alpha : float
        Step size parameter for the gradient updates
    use_baseline : bool
        Whether to use a reward baseline in the updates

    Returns:
    --------
    rewards : numpy.ndarray
        Average rewards over all bandits at each time step
    optimal_actions : numpy.ndarray
        Percentage of optimal actions over all bandits at each time step
    """
    num_bandits = len(bandits)
    num_steps = bandits[0].max_steps

    # Arrays to store results
    all_rewards = np.zeros((num_bandits, num_steps))
    all_optimal_actions = np.zeros((num_bandits, num_steps))

    # Run the algorithm on each bandit
    for i, bandit in enumerate(bandits):
        # Initialize the environment
        observation = bandit.start()

        # Initialize action preferences to zeros
        action_preferences = np.zeros(bandit.k)

        # Initialize baseline
        baseline = 0

        # Run steps
        for step in range(num_steps):
            # Softmax probability calculation
            exp_pref = np.exp(action_preferences)
            action_probs = exp_pref / np.sum(exp_pref)
            action = np.random.choice(bandit.k, p=action_probs)

            # Take action using the RLGlue step() method
            reward, next_observation, terminal, info = bandit.step(action)

            # Update baseline (average reward)
            if use_baseline:
                if step == 0:
                    baseline = reward
                else:
                    baseline += 0.1 * (reward - baseline)
            else:
                baseline = 0  # No baseline

            # Update action preferences using the gradient bandit update rule
            for a in range(bandit.k):
                if a == action:
                    # The selected action
                    action_preferences[a] += alpha * (reward - baseline) * (1 - action_probs[a])
                else:
                    # All other actions
                    action_preferences[a] -= alpha * (reward - baseline) * action_probs[a]

            # Store results
            all_rewards[i, step] = reward
            all_optimal_actions[i, step] = info['is_optimal']

            # Update observation for next step
            observation = next_observation

            # Check if terminal
            if terminal:
                break

    # Calculate average rewards and optimal action percentage across all bandits
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal_actions = np.mean(all_optimal_actions, axis=0)

    return avg_rewards, avg_optimal_actions


def run_single_configuration(args):
    """Worker function to run a single configuration."""
    testbed, config = args
    print(f"Running {config['name']}...")
    rewards, optimal_actions = run_gradient_algorithm(
        testbed,
        alpha=config['alpha'],
        use_baseline=config['baseline']
    )
    print(f"  Final average reward: {rewards[-1]:.4f}")
    print(f"  Final optimal action %: {optimal_actions[-1] * 100:.2f}%")
    return rewards, optimal_actions


def main():
    """
    Run the gradient bandit experiment to replicate Figure 2.5 in the book.
    """
    # Parameters for the testbed
    num_bandits = 2000
    k = 10
    runs = 1000
    seed = 42

    # Mean of +4 instead of 0 for the true action values
    mean = 4.0
    variance = 1.0

    print("Creating 10-armed testbed with mean +4...")
    testbed = KArmedBandit.create_testbed(
        num_bandits=num_bandits,
        k=k,
        runs=runs,
        mean=mean,  # Set mean to +4
        variance=variance,
        seed=seed
    )
    print("Testbed created!")

    # Define configurations to test
    configurations = [
        {'name': 'α=0.1, with baseline', 'alpha': 0.1, 'baseline': True},
        {'name': 'α=0.1, without baseline', 'alpha': 0.1, 'baseline': False},
        {'name': 'α=0.4, with baseline', 'alpha': 0.4, 'baseline': True},
        {'name': 'α=0.4, without baseline', 'alpha': 0.4, 'baseline': False}
    ]

    # Prepare arguments for parallel processing
    args = [(testbed, config) for config in configurations]

    # Run configurations in parallel
    with Pool() as pool:
        results = pool.map(run_single_configuration, args)

    # Unpack results
    all_rewards = [r[0] for r in results]
    all_optimal_actions = [r[1] for r in results]

    # Plot results
    plt.figure(figsize=(10, 6))

    # Define colors
    colors = {
        'baseline': ['#0100fb', '#8faafe'],      # blue, light blue
        'no_baseline': ['#986431', '#986431']    # brown, light brown
    }

    # Plot optimal action percentage with specific colors
    for i, config in enumerate(configurations):
        color_key = 'baseline' if config['baseline'] else 'no_baseline'
        color_idx = 1 if config['alpha'] == 0.4 else 0
        line = plt.plot(all_optimal_actions[i] * 100,
                       color=colors[color_key][color_idx],
                       linewidth=1)[0]

    # Calculate annotation positions
    x_alpha_01, x_alpha_04, x_baseline = 300, 900, 500  # Changed x_alpha_04 to 900

    # Add alpha annotations
    for config in configurations:
        x_pos = x_alpha_01 if config['alpha'] == 0.1 else x_alpha_04
        baseline_data = [d for d, c in zip(all_optimal_actions, configurations)
                        if c['baseline'] == config['baseline']]
        # Change vertical position based on alpha value
        offset = 5 if config['alpha'] == 0.1 else -5  # Positive for above, negative for below
        y_pos = baseline_data[0 if config['alpha'] == 0.1 else 1][x_pos] * 100 + offset
        va_pos = 'bottom' if config['alpha'] == 0.1 else 'top'  # Adjust vertical alignment
        color_key = 'baseline' if config['baseline'] else 'no_baseline'
        color_idx = 1 if config['alpha'] == 0.4 else 0
        plt.annotate(f'α = {config["alpha"]}', xy=(x_pos, y_pos),
                    ha='center', va=va_pos, fontsize=12,
                    color=colors[color_key][color_idx])

    # Add baseline annotations between their respective lines
    baseline_data = [d for d, c in zip(all_optimal_actions, configurations) if c['baseline']]
    no_baseline_data = [d for d, c in zip(all_optimal_actions, configurations) if not c['baseline']]

    baseline_y = (baseline_data[0][x_baseline] + baseline_data[1][x_baseline]) * 100 / 2
    no_baseline_y = (no_baseline_data[0][x_baseline] + no_baseline_data[1][x_baseline]) * 100 / 2

    plt.annotate('with baseline', xy=(x_baseline, baseline_y),
                ha='center', va='center', fontsize=12,
                color=colors['baseline'][0])  # blue
    plt.annotate('without baseline', xy=(x_baseline, no_baseline_y),
                ha='center', va='center', fontsize=12,
                color=colors['no_baseline'][0])  # brown

    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action', rotation=0, ha='right', va='center', labelpad=15)
    plt.title('Figure 2.5: Gradient Bandit Algorithm with True Values ~ N(4,1)')

    # Set y-axis properties
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20), [f'{int(x)}%' for x in np.arange(0, 101, 20)])

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove grid
    plt.grid(False)

    plt.tight_layout()
    plt.savefig('gradient_bandit_results.png')
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for the 10-armed Testbed experiment from Sutton & Barto's Reinforcement Learning book.
This script replicates the experiment described in Section 2.3 of the book.
"""

import matplotlib.pyplot as plt
import numpy as np

from environments.KArmedBandit import KArmedBandit


def run_algorithm(bandits, algorithm, epsilon=0.0, alpha=None, optimistic_init=0.0, ucb_c=None):
    """
    Run a bandit algorithm on a set of bandit problems.

    Parameters:
    -----------
    bandits : list
        List of KArmedBandit environments
    algorithm : str
        'epsilon-greedy', 'greedy', 'ucb', or 'gradient'
    epsilon : float
        Exploration rate for epsilon-greedy
    alpha : float
        Step size for non-stationary updates; if None, use sample average
    optimistic_init : float
        Initial optimistic value for action-value estimates
    ucb_c : float
        Exploration parameter for UCB algorithm

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
        # Initialize the environment using the RLGlue start() method
        observation = bandit.start()

        # Initialize action values with optimistic initialization if specified
        action_values = np.ones(bandit.k) * optimistic_init
        action_counts = np.zeros(bandit.k)

        # For the gradient bandit algorithm
        if algorithm == 'gradient':
            action_preferences = np.zeros(bandit.k)
            baseline = 0

        # Run steps
        for step in range(num_steps):
            # Select action based on the algorithm
            if algorithm == 'epsilon-greedy' or algorithm == 'greedy':
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(bandit.k)  # Explore
                else:
                    action = np.argmax(action_values)  # Exploit

            elif algorithm == 'ucb':
                # UCB action selection
                t = step + 1  # time step (adding 1 to avoid division by zero)
                if 0 in action_counts:
                    # Choose first action that hasn't been tried yet
                    action = np.where(action_counts == 0)[0][0]
                else:
                    # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
                    exploration = np.sqrt(np.log(t) / action_counts)
                    ucb_values = action_values + ucb_c * exploration
                    action = np.argmax(ucb_values)

            elif algorithm == 'gradient':
                # Softmax probability calculation
                exp_pref = np.exp(action_preferences)
                action_probs = exp_pref / np.sum(exp_pref)
                action = np.random.choice(bandit.k, p=action_probs)

            # Take action using the RLGlue step() method
            # Returns: reward, next_observation, terminal, info
            reward, next_observation, terminal, info = bandit.step(action)

            # Update action values based on algorithm
            if algorithm in ['epsilon-greedy', 'greedy', 'ucb']:
                action_counts[action] += 1
                # Use constant step-size if alpha is provided, otherwise sample average
                if alpha is not None:
                    action_values[action] += alpha * (reward - action_values[action])
                else:
                    action_values[action] += (reward - action_values[action]) / action_counts[action]

            elif algorithm == 'gradient':
                # Update baseline (average reward)
                if step == 0:
                    baseline = reward
                else:
                    baseline += 0.1 * (reward - baseline)

                # Update action preferences
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


def main():
    """
    Run the 10-armed testbed experiment with different algorithms and parameters.
    """
    # Parameters for the testbed
    num_bandits = 2000
    k = 10
    runs = 1000
    seed = 42  # For reproducibility

    print("Creating 10-armed testbed with {} bandits...".format(num_bandits))
    testbed = KArmedBandit.create_testbed(
        num_bandits=num_bandits,
        k=k,
        runs=runs,
        seed=seed
    )
    print("Testbed created!")

    # Define algorithms to test
    algorithms = [
        {'name': 'Greedy (ε=0)', 'algo': 'greedy', 'epsilon': 0.0},
        {'name': 'ε-greedy (ε=0.1)', 'algo': 'epsilon-greedy', 'epsilon': 0.1},
        {'name': 'ε-greedy (ε=0.01)', 'algo': 'epsilon-greedy', 'epsilon': 0.01}
    ]

    # Arrays to store results
    all_rewards = []
    all_optimal_actions = []

    # Run each algorithm
    for alg in algorithms:
        print(f"Running {alg['name']}...")
        rewards, optimal_actions = run_algorithm(
            testbed,
            alg['algo'],
            epsilon=alg.get('epsilon', 0.0)
        )

        all_rewards.append(rewards)
        all_optimal_actions.append(optimal_actions)

        # Print some statistics
        print(f"  Final average reward: {rewards[-1]:.4f}")
        print(f"  Final optimal action %: {optimal_actions[-1] * 100:.2f}%")
        print()

    # Plot results
    plt.figure(figsize=(18, 6))

    # Plot average rewards
    plt.subplot(1, 2, 1)
    for i, alg in enumerate(algorithms):
        plt.plot(all_rewards[i], label=alg['name'])
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()
    plt.grid(True)

    # Plot optimal action percentage
    plt.subplot(1, 2, 2)
    for i, alg in enumerate(algorithms):
        plt.plot(all_optimal_actions[i] * 100, label=alg['name'])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Optimal Action Percentage over Time')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('10_armed_testbed_results.png')
    plt.show()


if __name__ == "__main__":
    main()

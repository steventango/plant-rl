import sys
import os

# Add the src directory to the path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import numpy as np
from algorithms.nn.GreedyAC.policy.MLP import Dirichlet, Gaussian


def test_dirichlet_vs_gaussian_shapes():
    """
    Test that compares the output shapes of log_prob and sample methods
    for Dirichlet and Gaussian policy classes.
    """
    # Common parameters
    input_dim = 10
    action_dim = 5
    hidden_dim = 64
    n_hidden = 2
    activation = "relu"
    init = "xavier_uniform"
    batch_size = 32
    num_samples = 1
    
    # Create dummy state batch
    state_batch = torch.randn(batch_size, input_dim)
    
    # Initialize Dirichlet policy
    dirichlet_policy = Dirichlet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation=activation,
        offset=0.0,
        init=init,
    )
    
    # Initialize Gaussian policy
    action_min = 0.0
    action_max = 1.0
    gaussian_policy = Gaussian(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation=activation,
        clip_stddev=1000,
        init=init,
        action_min=action_min,
        action_max=action_max,
    )
    
    print("=" * 80)
    print("Testing sample() method")
    print("=" * 80)
    
    # Test sample method for Dirichlet
    print("\nDirichlet Policy - sample():")
    dir_action, dir_log_prob, dir_mean = dirichlet_policy.sample(state_batch, num_samples=num_samples)
    print(f"  Action shape: {dir_action.shape}")
    print(f"  Log prob shape: {dir_log_prob.shape}")
    print(f"  Mean shape: {dir_mean.shape}")
    print(f"  Action sample:\n{dir_action[0]}")
    print(f"  Action sum (should be ~1.0): {dir_action[0].sum().item()}")
    
    # Test sample method for Gaussian
    print("\nGaussian Policy - sample():")
    gauss_action, gauss_log_prob, gauss_mean = gaussian_policy.sample(state_batch, num_samples=num_samples)
    print(f"  Action shape: {gauss_action.shape}")
    print(f"  Log prob shape: {gauss_log_prob.shape}")
    print(f"  Mean shape: {gauss_mean.shape}")
    print(f"  Action sample:\n{gauss_action[0]}")
    
    print("\n" + "=" * 80)
    print("Testing rsample() method")
    print("=" * 80)
    
    # Test rsample method for Dirichlet
    print("\nDirichlet Policy - rsample():")
    dir_action_r, dir_log_prob_r, dir_mean_r = dirichlet_policy.rsample(state_batch, num_samples=num_samples)
    print(f"  Action shape: {dir_action_r.shape}")
    print(f"  Log prob shape: {dir_log_prob_r.shape}")
    print(f"  Mean shape: {dir_mean_r.shape}")
    
    # Test rsample method for Gaussian
    print("\nGaussian Policy - rsample():")
    gauss_action_r, gauss_log_prob_r, gauss_mean_r = gaussian_policy.rsample(state_batch, num_samples=num_samples)
    print(f"  Action shape: {gauss_action_r.shape}")
    print(f"  Log prob shape: {gauss_log_prob_r.shape}")
    print(f"  Mean shape: {gauss_mean_r.shape}")
    
    print("\n" + "=" * 80)
    print("Testing log_prob() method")
    print("=" * 80)
    
    # Test log_prob method for Dirichlet
    print("\nDirichlet Policy - log_prob():")
    # Normalize actions to be valid for Dirichlet (sum to 1)
    dir_actions = torch.rand(batch_size, action_dim)
    dir_actions = dir_actions / dir_actions.sum(dim=-1, keepdim=True)
    dir_log_prob_computed = dirichlet_policy.log_prob(state_batch, dir_actions)
    print(f"  Log prob shape: {dir_log_prob_computed.shape}")
    print(f"  Log prob sample values: {dir_log_prob_computed[:3]}")
    
    # Test log_prob method for Gaussian
    print("\nGaussian Policy - log_prob():")
    gauss_actions = torch.rand(batch_size, action_dim) * (action_max - action_min) + action_min
    gauss_log_prob_computed = gaussian_policy.log_prob(state_batch, gauss_actions)
    print(f"  Log prob shape: {gauss_log_prob_computed.shape}")
    print(f"  Log prob sample values: {gauss_log_prob_computed[:3]}")
    
    print("\n" + "=" * 80)
    print("Testing with multiple samples")
    print("=" * 80)
    
    num_samples_multi = 10
    
    # Test Dirichlet with multiple samples
    print(f"\nDirichlet Policy - sample(num_samples={num_samples_multi}):")
    dir_action_multi, dir_log_prob_multi, dir_mean_multi = dirichlet_policy.sample(state_batch, num_samples=num_samples_multi)
    print(f"  Action shape: {dir_action_multi.shape}")
    print(f"  Log prob shape: {dir_log_prob_multi.shape}")
    print(f"  Mean shape: {dir_mean_multi.shape}")
    
    # Test Gaussian with multiple samples
    print(f"\nGaussian Policy - sample(num_samples={num_samples_multi}):")
    gauss_action_multi, gauss_log_prob_multi, gauss_mean_multi = gaussian_policy.sample(state_batch, num_samples=num_samples_multi)
    print(f"  Action shape: {gauss_action_multi.shape}")
    print(f"  Log prob shape: {gauss_log_prob_multi.shape}")
    print(f"  Mean shape: {gauss_mean_multi.shape}")
    
    print("\n" + "=" * 80)
    print("Shape comparison summary")
    print("=" * 80)
    
    print("\nFor single sample (num_samples=1):")
    print(f"  Dirichlet action shape: {dir_action.shape} | Gaussian action shape: {gauss_action.shape}")
    print(f"  Dirichlet log_prob shape: {dir_log_prob.shape} | Gaussian log_prob shape: {gauss_log_prob.shape}")
    print(f"  Dirichlet mean shape: {dir_mean.shape} | Gaussian mean shape: {gauss_mean.shape}")
    
    print(f"\nFor multiple samples (num_samples={num_samples_multi}):")
    print(f"  Dirichlet action shape: {dir_action_multi.shape} | Gaussian action shape: {gauss_action_multi.shape}")
    print(f"  Dirichlet log_prob shape: {dir_log_prob_multi.shape} | Gaussian log_prob shape: {gauss_log_prob_multi.shape}")
    print(f"  Dirichlet mean shape: {dir_mean_multi.shape} | Gaussian mean shape: {gauss_mean_multi.shape}")
    
    print("\nFor log_prob() with action input:")
    print(f"  Dirichlet log_prob shape: {dir_log_prob_computed.shape} | Gaussian log_prob shape: {gauss_log_prob_computed.shape}")
    
    print("\n" + "=" * 80)
    print("Tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_dirichlet_vs_gaussian_shapes()

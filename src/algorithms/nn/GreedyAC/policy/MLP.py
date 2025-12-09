# Import modules  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.distributions import Independent, Normal

from ..utils.nn_utils import weights_init_

# Global variables
EPSILON = 1e-6

#TODO Remove action_space from all policy classes, replace with action_dim and range.

class SquashedGaussian(nn.Module):
    """
    Class SquashedGaussian implements a policy following a squashed
    Gaussian distribution in each state, parameterized by an MLP.
    """

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        n_hidden,
        activation,
        action_space=None,
        clip_stddev=1000,
        init=None,
    ):
        """
        Constructor

        Parameters
        ----------
        input_dim : int
            The number of elements in the state feature vector
        action_dim : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        activation : str
            The activation function to use, one of 'relu', 'tanh'
        action_space : gym.spaces.Space, optional
            The action space of the environment, by default None. This argument
            is used to ensure that the actions are within the correct scale.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(SquashedGaussian, self).__init__()

        self.action_dim = action_dim

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        # Set up the layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))  # type: ignore

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.act(layer(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        if self.clip_stddev:
            log_std = torch.clamp(
                log_std, min=-self.clip_std_threshold, max=self.clip_std_threshold
            )
        return mean, log_std

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.action_dim > 1:
            normal = Independent(normal, 1)

        x_t = normal.sample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= (
            torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)  # type: ignore
            .sum(axis=-1)
            .reshape(log_prob.shape)
        )
        if self.action_dim > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state using
        the reparameterization trick

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.action_dim > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        x_t = normal.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= (
            torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)  # type: ignore
            .sum(axis=-1)
            .reshape(log_prob.shape)
        )
        if self.action_dim > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

    def log_prob(self, state_batch, x_t_batch):
        """
        Calculates the log probability of taking the action generated
        from x_t, where x_t is returned from sample or rsample. The
        log probability is returned for each action dimension separately.
        """
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.action_dim > 1:
            normal = Independent(normal, 1)

        y_t = torch.tanh(x_t_batch)
        log_prob = normal.log_prob(x_t_batch)
        log_prob -= (
            torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)  # type: ignore
            .sum(axis=-1)
            .reshape(log_prob.shape)
        )
        if self.action_dim > 1:
            log_prob = log_prob.unsqueeze(-1)

        return log_prob

    def to(self, device):  # type: ignore
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SquashedGaussian, self).to(device)


class Softmax(nn.Module):
    """
    Softmax implements a softmax policy in each state, parameterized
    using an MLP to predict logits.
    """

    def __init__(
        self, input_dim, action_dim, hidden_dim, n_hidden, activation, init=None
    ):
        super(Softmax, self).__init__()

        self.action_dim = action_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        # self.apply(weights_init_)
        self.apply(lambda module: weights_init_(module, init, activation))  # type: ignore

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

        print("Actor All submodules (including nested ones):")
        for module in self.modules():
            print(module)

    def forward(self, state):
        x = self.act(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.output_layer(x)
        return x

    def sample(self, state, num_samples=1):
        logits = self.forward(state)

        if len(logits.shape) != 1 and (
            len(logits.shape) != 2 and 1 not in logits.shape
        ):
            shape = logits.shape
            raise ValueError(f"expected a vector of logits, got shape {shape}")

        probs = F.softmax(logits, dim=1)

        policy = torch.distributions.Categorical(probs)
        actions = policy.sample((num_samples,))

        log_prob = F.log_softmax(logits, dim=1)

        log_prob = torch.gather(log_prob, dim=1, index=actions)
        if num_samples == 1:
            actions = actions.squeeze(0)
            log_prob = log_prob.squeeze(0)

        actions = actions.unsqueeze(-1)
        log_prob = log_prob.unsqueeze(-1)

        # return actions.float(), log_prob, None
        return actions.int(), log_prob, logits.argmax(dim=-1)

    def all_log_prob(self, states):
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs

    def log_prob(self, states, actions):
        """
        Returns the log probability of taking actions in states.
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = torch.gather(log_probs, dim=1, index=actions.long())

        return log_probs

    def to(self, device):  # type: ignore
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        return super(Softmax, self).to(device)


class Gaussian(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)` using a `tanh` activation.
    """

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        n_hidden,
        activation,
        action_space,
        clip_stddev=1000,
        init=None,
    ):
        """
        Constructor

        Parameters
        ----------
        input_dim : int
            The number of elements in the state feature vector
        action_dim : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(Gaussian, self).__init__()

        self.action_dim = action_dim

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))  # type: ignore

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.act(layer(x))

        mean = torch.tanh(self.mean_linear(x))
        mean = ((mean + 1) / 2) * (
            self.action_max - self.action_min
        ) + self.action_min  # ∈ [action_min, action_max]
        log_std = self.log_std_linear(x)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(
                log_std, min=-self.clip_std_threshold, max=self.clip_std_threshold
            )
        return mean, log_std

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.action_dim > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        action = normal.rsample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        return action, log_prob, mean

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.action_dim > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        # print(action.shape)

        return action, log_prob, mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.action_dim > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)
        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])  # type: ignore

        return log_prob

    def to(self, device):  # type: ignore
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(Gaussian, self).to(device)



class Dirichlet(nn.Module):
    """
    Class Dirichlet implements a policy following Dirichlet distribution
    in each state, parameterized as an MLP.
    """

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        n_hidden,
        activation,
        offset=0.0,
        init=None,
    ):
        """
        Constructor

        Parameters
        ----------
        input_dim : int
            The number of elements in the state feature vector
        action_dim : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(Dirichlet, self).__init__()

        self.action_dim = action_dim

        self.epsilon = 1e-2
        self.clip_alpha = 15.0
        self.offset = (
            offset + 1e-5
        )  # the small value prevents distrax.Dirichlet from having nan mode

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        
        self.alpha_layer = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))  # type: ignore

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Forward pass (the returned action is only used in deterministic mode)
        """
        alpha = self.get_alpha(state)
        pi_distribution = distributions.Dirichlet(concentration=alpha)
        pi_mean = pi_distribution.mode()
        pi_mean = torch.where(
            torch.isnan(pi_mean),
            pi_distribution.mean,
            pi_mean,
        )
        return pi_mean, pi_distribution

    def clip(self, action):
        clipped_action = torch.clamp(action, self.epsilon, 1.0 - self.epsilon)
        clipped_action = clipped_action / torch.sum(
            clipped_action, dim=-1, keepdim=True
        )
        return clipped_action

    def get_alpha(self, state):
        x = self.act(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        alpha_logits = self.alpha_layer(x)
        alpha = torch.sigmoid(alpha_logits) * self.clip_alpha + self.offset
        return alpha
        
    def rsample(self, state, num_samples=1):
        pi_mean, pi_distribution = self.forward(state)
        pi_action = pi_distribution.rsample((num_samples,))

        pi_action = self.clip(pi_action)
        pi_mean = self.clip(pi_mean)
        
        if num_samples == 1:
            pi_action = pi_action.squeeze(0)

        log_prob = pi_distribution.log_prob(pi_action)
        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        # print(action.shape)

        return pi_action, log_prob, pi_mean

    def sample(self, state, num_samples=1):
        pi_mean, pi_distribution = self.forward(state)
        pi_action = pi_distribution.sample((num_samples,))

        pi_action = self.clip(pi_action)
        pi_mean = self.clip(pi_mean)
        
        if num_samples == 1:
            pi_action = pi_action.squeeze(0)

        log_prob = pi_distribution.log_prob(pi_action)
        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        # print(action.shape)

        return pi_action, log_prob, pi_mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        alpha = self.get_alpha(states)
        pi_distribution = distributions.Dirichlet(concentration=alpha)
        clipped_actions = self.clip(actions)
        log_prob = pi_distribution.log_prob(clipped_actions)

        if self.action_dim == 1:
            log_prob.unsqueeze(-1)

        return log_prob

    def to(self, device):  # type: ignore
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.epsilon = self.epsilon.to(device)
        self.clip_alpha = self.clip_alpha.to(device)
        self.offset = self.offset.to(device)
        return super(Dirichlet, self).to(device)



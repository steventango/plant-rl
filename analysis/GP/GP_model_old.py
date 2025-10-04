import numpy as np
import gpjax as gpx
import jax.numpy as jnp
import jax
from jax import random

jax.config.update("jax_enable_x64", True)


class GP:
    def __init__(
        self,
        data,
        kernel=gpx.kernels.RBF(),
        meanf=gpx.mean_functions.Zero(),
        action_traces_weights=jnp.array([0.7, 0.5, 0.3]),
        key=random.PRNGKey(0),
    ):
        self.kernel = kernel
        self.meanf = meanf

        self.trace_2nd_dim = action_traces_weights.shape[0]
        self.action_traces_weights = jnp.tile(action_traces_weights, (3, 1)).T

        self.key = key
        self.temp = jnp.array(0.3)

        self.preprocess(data)
        self.learn()

    def preprocess(self, data):
        """
        preprocess the data to be given to the GP
        normalize the state (both input and target state)
        modify the actions to be binary arrays
        compute the trace of the actions
        seperate out train and test splits

        returns cleaned data as a gpx dataset, meta data (normalization data, trace weights)
        """

        # compute delta: delta = y - x
        delta = data[:, -1] - data[:, 0]

        # normalize x and deltas
        self.input_mean = data[:, 0].mean()
        self.input_std = data[:, 0].std()

        self.delta_mean = delta.mean()
        self.delta_std = delta.std()

        X_train = []
        Y_train = []

        trace = jnp.zeros((3, self.trace_2nd_dim))

        for i in range(len(data) - 1):
            normalized_state = (data[i, 0] - self.input_mean) / self.input_std
            normalized_delta = (delta[i] - self.delta_mean) / self.delta_std

            action = jax.nn.one_hot(data[i, 1] + 1, 3)
            trace = (
                self.action_traces_weights * jnp.tile(action, (3, 1))
                + (1 - self.action_traces_weights) * trace
            )

            clean_x = jnp.concat(
                (jnp.array([normalized_state]), action, trace.flatten())
            )
            clean_y = jnp.array([normalized_delta])

            X_train.append(clean_x)
            Y_train.append(clean_y)

            if (
                data[i, -1].item() != data[i + 1, 0].item()
            ):  # check if the next point is in sequence with this point
                trace = jnp.zeros((3, self.trace_2nd_dim))

        normalized_state = (data[-1, 0] - self.input_mean) / self.input_std
        normalized_delta = (delta[-1] - self.delta_mean) / self.delta_std

        action = jax.nn.one_hot(data[-1, 1] + 1, 3)
        trace = (
            self.action_traces_weights * jnp.tile(action, (3, 1))
            + (1 - self.action_traces_weights) * trace
        )

        clean_x = jnp.concat((jnp.array([normalized_state]), action, trace.flatten()))
        clean_y = jnp.array([normalized_delta])

        X_train.append(clean_x)
        Y_train.append(clean_y)

        X_train = jnp.array(X_train)
        Y_train = jnp.array(Y_train)

        self.D = gpx.Dataset(X=X_train, y=Y_train)

    def learn(self):
        prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)
        posterior = prior * likelihood
        self.opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.D,
            trainable=gpx.parameters.Parameter,
        )

    def get_predictive_dist(self, x):
        latent_dist = self.opt_posterior.predict(x, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        return predictive_dist

    def predict_mean_std(self, x):
        predictive_dist = self.get_predictive_dist(x)
        predictive_mean = predictive_dist.mean
        predictive_std = jnp.sqrt(predictive_dist.variance)
        return predictive_mean, predictive_std

    def sample_deltas(self, x, key=random.PRNGKey(0), N=30):
        # normalize x
        # X_train.at[:, 0].set((X_train[:, 0] - model.input_mean) / model.input_std)
        x = x.at[:, 0].set((x[:, 0] - self.input_mean) / self.input_std)
        # get normalized delta mean and std
        mean, std = self.predict_mean_std(x)
        # get normalized samples of delta
        normalized_delta_samples = (
            random.normal(key, (mean.shape[0], N)) * std[:, None]
        ) + mean[:, None]
        # unnormalize samples of delta
        delta_samples = (normalized_delta_samples * self.delta_std) + self.delta_mean
        return delta_samples

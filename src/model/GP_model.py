import gpjax as gpx
import jax.numpy as jnp
import jax
from jax import random

jax.config.update("jax_enable_x64", True)


class GP:
    def __init__(
        self,
        input_data,
        output_data,
        kernel = None,
        meanf = None,
        key = None,
    ):
        self.kernel = kernel if kernel is not None else gpx.kernels.Matern52()
        self.meanf = meanf if meanf is not None else gpx.mean_functions.Zero()
        self.key = key if key is not None else random.PRNGKey(0)
        X_train, self.input_mean, self.input_std = self.preprocess(input_data)
        Y_train, self.output_mean, self.output_std = self.preprocess(output_data)
        self.D = gpx.Dataset(X=X_train, y=Y_train)
        self.learn()

    def preprocess(self, data):
        # normalize every input dimension
        norm_data = []
        mean = []
        std = []
        for i in range(data.shape[1]):
            mean.append(data[:, i].mean())
            std.append(data[:, i].std())
            col = []
            for j in range(data.shape[0]):
                normalized_state = (data[j, i] - mean[-1]) / std[-1]
                col.append(normalized_state)
            norm_data.append(col)

        return jnp.vstack(norm_data).T, jnp.array(mean), jnp.array(std)

    def normalize_input(self, x):
        return (jnp.array(x) - self.input_mean) / self.input_std

    def denormalize_output(self, x):
        return jnp.array(x) * self.output_std + self.output_mean

    def learn(self):
        prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)
        posterior = prior * likelihood
        self.opt_posterior, _ = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),  # type: ignore
            train_data=self.D,
        )

    def get_predictive_dist(self, X):
        latent_dist = self.opt_posterior.predict(X, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        return predictive_dist

    def predict_mean_std(self, X):
        X = jnp.vstack([self.normalize_input(x) for x in X])
        predictive_dist = self.get_predictive_dist(X)
        predictive_mean = predictive_dist.mean()
        predictive_std = jnp.sqrt(predictive_dist.variance())   # type: ignore
        return self.denormalize_output(predictive_mean), jnp.array(
            predictive_std
        ) * self.output_std

    def sample_output(self, x, N=100):
        mean, std = self.predict_mean_std(x)
        self.key, subkey = random.split(self.key)
        samples = (random.normal(subkey, (mean.shape[0], N)) * std[:, None]) + mean[
            :, None
        ]
        return samples

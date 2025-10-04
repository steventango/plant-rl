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
        kernel=gpx.kernels.Matern52(),
        meanf=gpx.mean_functions.Zero(),
        key=random.PRNGKey(0),
    ):
        self.kernel = kernel
        self.meanf = meanf
        self.key = key
        X_train, self.input_mean, self.input_std = self.preprocess(input_data)
        Y_train, self.output_mean, self.output_std = self.preprocess(output_data)
        self.D = gpx.Dataset(X=X_train, y=Y_train)
        self.learn()

    def preprocess(self, data):
        # normalize every intput dimension
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

    def learn(self):
        prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)
        posterior = prior * likelihood
        self.opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.D,
        )

    def get_predictive_dist(self, x):
        latent_dist = self.opt_posterior.predict(x, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        return predictive_dist

    def predict_mean_std(self, x):
        predictive_dist = self.get_predictive_dist(x)
        predictive_mean = predictive_dist.mean()
        predictive_std = jnp.sqrt(predictive_dist.variance())
        return predictive_mean, predictive_std

    def sample_output(self, x, N=100):
        x = (x - self.input_mean) / self.input_std
        mean, std = self.predict_mean_std(x.reshape(1, -1))
        normalized_samples = (
            random.normal(self.key, (mean.shape[0], N)) * std[:, None]
        ) + mean[:, None]
        normalized_samples = normalized_samples.T
        samples = [
            sample * self.output_std + self.output_mean for sample in normalized_samples
        ]
        return samples

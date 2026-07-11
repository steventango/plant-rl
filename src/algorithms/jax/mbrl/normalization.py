import jax.numpy as jnp
from flax import nnx


class NormalizeVecObs(nnx.Module):
    def __init__(self, x, use_running_average: bool = False, eps: float = 1e-8):
        super().__init__()
        self.use_running_average = use_running_average
        self.eps = eps
        self.mean = nnx.Variable(jnp.zeros_like(x))
        self.var = nnx.Variable(jnp.ones_like(x))
        self.count = nnx.Variable(jnp.array(1e-4))

    def __call__(self, x):
        if self.use_running_average:
            return self.normalize(x)

        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean[...] += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / total_count
        self.var[...] = M2 / total_count
        self.count[...] = total_count
        return self.normalize(x)

    def normalize(self, x):
        return (x - self.mean) / jnp.sqrt(self.var + self.eps)

    def denormalize(self, x):
        return x * jnp.sqrt(self.var + self.eps) + self.mean


class NormalizeVecReward(nnx.Module):
    def __init__(
        self, reward, gamma, use_running_average: bool = False, eps: float = 1e-8
    ):
        super().__init__()
        self.use_running_average = use_running_average
        self.eps = eps
        self.gamma = gamma
        batch_count = reward.shape[0]
        self.mean = nnx.Variable(jnp.zeros(1))
        self.var = nnx.Variable(jnp.ones(1))
        self.count = nnx.Variable(jnp.array(1e-4))
        self.return_val = nnx.Variable(jnp.zeros(batch_count))

    def __call__(self, reward, terminated, truncated):
        if self.use_running_average:
            return self.normalize(reward)

        done = terminated | truncated
        return_val = self.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = reward.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean[...] += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / total_count
        self.var[...] = M2 / total_count
        self.count[...] = total_count
        self.return_val[...] = return_val
        return self.normalize(reward)

    def normalize(self, reward):
        return reward / jnp.sqrt(self.var + self.eps)

    def denormalize(self, reward):
        return reward * jnp.sqrt(self.var + self.eps)

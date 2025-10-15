from flax import nnx
from jax.nn import initializers


class FCBody(nnx.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=(64, 64),
        activation=nnx.relu,
        init_type='xavier',
        info=None,
        *,
        rngs: nnx.Rngs,
    ):
        dims = (input_dim,) + hidden_units

        kernel_init = None
        bias_init = None

        if init_type == "xavier":
            kernel_init = initializers.xavier_uniform()
            bias_init = initializers.zeros
        elif init_type == "uniform":
            kernel_init = initializers.uniform(scale=0.003)
            bias_init = initializers.zeros
        elif init_type == "zeros":
            kernel_init = initializers.zeros
            bias_init = initializers.zeros
        elif init_type == "constant":
            if info is None:
                raise ValueError("Constant value 'info' must be provided for init_type 'constant'")
            kernel_init = initializers.constant(info)
            bias_init = initializers.constant(info)
        else:
            raise ValueError(f'init_type is not defined: {init_type}')

        self.layers = [
            nnx.Linear(
                dim_in,
                dim_out,
                kernel_init=kernel_init,
                bias_init=bias_init,
                rngs=rngs,
            )
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ]
        self.activation = activation
        self.feature_dim = dims[-1]

    def __call__(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

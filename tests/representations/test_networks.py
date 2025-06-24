import haiku as hk
import jax.numpy as jnp
from representations.networks import NetworkBuilder

def one_layer_Relu():
    builder = NetworkBuilder(
        input_shape=(2,),
        params={
            "hidden": 32,
            "type": 'OneLayerRelu',
        },
        seed=0,
    )

    actions = 2
    feature_function = builder.getFeatureFunction()

    q_function = builder.addHead(lambda: hk.Linear(actions, name='q'))
    params = builder.getParams()

    x = jnp.zeros((1, 2))
    phi = feature_function(params, x)
    assert phi.activations["phi"].shape == (1, 32)
    q = q_function(params, phi.out)
    assert q.shape == (1, actions)

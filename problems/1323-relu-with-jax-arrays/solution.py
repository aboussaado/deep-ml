import jax
import jax.numpy as jnp

def relu(x):
    """Return x with negative entries replaced by 0 (same shape as x)."""
    x = jnp.array(x)
    return jax.nn.relu(x)

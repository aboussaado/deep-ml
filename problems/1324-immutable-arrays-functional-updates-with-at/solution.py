import jax
import jax.numpy as jnp

def replace_at(x, indices, values):
    """Return a copy of 1-D array x with x[indices] replaced by values.
    Must not modify x."""
    y = jnp.copy(x)
    for k in range(len(indices)):
        y = y.at[indices[k]].set(values[k])
    return y
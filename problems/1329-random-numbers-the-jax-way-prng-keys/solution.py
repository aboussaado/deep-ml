import jax
import jax.numpy as jnp

def sample_pair(seed, shape):
    """Return (a, b): two different standard-normal arrays of `shape`,
    drawn from two subkeys split off PRNGKey(seed)."""
    key = jax.random.PRNGKey(seed)
    sub1, sub2 = jax.random.split(key, 2)
    
    return jax.random.normal(sub1, shape), jax.random.normal(sub2, shape)
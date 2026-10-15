import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    def f(u): return u**3 + 2 * u
    grad = jax.grad(f)
    return float(grad(x))


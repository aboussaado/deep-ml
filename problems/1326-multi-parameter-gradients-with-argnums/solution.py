import jax
import jax.numpy as jnp

def mse_grads(w, b, x, y):
    """Return (dL/dw, dL/db) for L = mean((w*x + b - y)**2) via jax.grad.
    w, b are floats; x, y are 1-D jnp arrays. Returns two floats."""
    def L(w,b): return jnp.mean((w*x + b - y)**2)
    dL = jax.grad(L, argnums=(0, 1))
    dL_dw, dL_db = dL(w,b)
    return float(dL_dw), float(dL_db)
import jax
import jax.numpy as jnp

def per_sample_grads(w, b, X, y):
    """Per-example gradients of (w·x + b - y)^2.
    Returns (dW, db) with shapes (N, D) and (N,)."""
    w = jnp.asarray(w, dtype=float)
    b = jnp.asarray(b, dtype=float)
    X = jnp.asarray(X, dtype=float)
    y = jnp.asarray(y, dtype=float)
    def li(w,b, x, yi): return (jnp.dot(w,x) + b - yi)**2
    grad = jax.vmap(jax.grad(li, argnums=(0, 1)), in_axes=(None, None, 0, 0))
    dW, db = grad(w, b, X, y)
    return dW, db
import jax
import jax.numpy as jnp

def fit_line(x, y, lr, steps):
    """Gradient-descent fit of y ≈ w*x + b from w=b=0.
    Returns (w, b, final_loss) as Python floats."""
    def loss(w,b): return jnp.mean((w*x + b - y)**2)
    w,b = 0.0, 0.0
    for i in range(steps):
        L, (dw, db) = jax.value_and_grad(loss,argnums=(0,1))(w,b)
        w = w - lr * dw
        b = b - lr * db
    return float(w),float(b),float(loss(w,b))
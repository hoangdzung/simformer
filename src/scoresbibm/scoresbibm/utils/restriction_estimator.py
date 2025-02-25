import jax.numpy as jnp
import torch
from typing import Any, Callable
import numpy as np 

def  get_jax_density_thresholder(
    dist: Any,
    quantile: float = 1e-4,
    num_samples_to_estimate_support: int = 1_000_000,
    x_o=None,
    rng=None,
    return_type: str = 'torch',
) -> Callable:

    samples = dist.sample(num_samples=num_samples_to_estimate_support, x_o=x_o, rng=rng)
    log_probs = dist.log_prob(samples)
    sorted_log_probs = jnp.sort(log_probs)
    log_prob_threshold = sorted_log_probs[int(quantile * num_samples_to_estimate_support)]

    def density_thresholder(theta: jnp.ndarray) -> Any:
        theta_log_probs = dist.log_prob(theta)
        result = theta_log_probs > log_prob_threshold
        if return_type == 'torch':
            return torch.from_numpy(np.array(result))
        elif return_type == 'jax':
            return result
        else:
            raise ValueError("return_type must be either 'torch' or 'jax'")

    return density_thresholder 
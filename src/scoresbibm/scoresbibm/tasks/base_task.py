

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from functools import partial

import torch
import jax 
import jax.numpy as jnp
from sbibm.tasks.simulator import Simulator



class Task(ABC):
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        self.name = name
        self.backend = backend    
        
    @property
    def theta_dim(self):
        return self.get_theta_dim()
    
    @property
    def x_dim(self):
        return self.get_x_dim()
    
    def get_theta_dim(self):
        raise NotImplementedError()
    
    def get_x_dim(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_node_id(self):
        raise NotImplementedError()
    
    def get_batch_sampler(self):
        return base_batch_sampler
    

    def get_base_mask_fn(self):
        raise NotImplementedError()
    
    
class InferenceTask(Task):
    
    observations = range(1, 11)
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_prior(self):
        raise NotImplementedError()
        
    def get_simulator(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation(self, index: int):
        raise NotImplementedError()
    
    def get_reference_posterior_samples(self, index: int):
        raise NotImplementedError()
    
    def get_true_parameters(self, index: int):
        raise NotImplementedError()
    
    
class AllConditionalTask(Task):
    
    var_names: list[str]
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_joint_sampler(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation_generator(self):
        raise NotImplementedError()
    
    def get_base_mask_fn(self):
        raise NotImplementedError()
        
    def get_reference_sampler(self):
        raise NotImplementedError()

class CGMTask(Task):
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()

    def get_test_params(self, index: int):
        raise NotImplementedError()

    def get_observations_from_params(self, params: torch.tensor):
        simulator = self.get_simulator()
        out = simulator(params)
        if self.dim_external > 0:
            out = out[:, : -self.dim_external]
        return out

    def get_observation(self, index: int):
        simulator = self.get_simulator()
        params = self.get_test_params(index)
        return simulator(params)


    def get_torch_prior(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist
    
    def get_prior_dist(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist

    def get_prior(self):
        if self.backend == "torch":
            return self.get_prior_dist()
        else:
            raise NotImplementedError()
        
    def get_prior_params(self) -> Dict:
        """Get parameters of prior distribution"""
        return self.prior_params

    def get_labels_data(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"data_{i+1}" for i in range(self.dim_data)]

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"parameter_{i+1}" for i in range(self.dim_parameters)]
    
    def get_theta_dim(self):
        return self.dim_parameters
    
    def get_x_dim(self):
        return self.dim_data

    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        return data.reshape(-1, self.dim_data)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data

        Tasks that require more than 2 dimensions for output of the simulator (e.g.
        returning images) may override this method.
        """
        return data.reshape(-1, self.dim_data)

partial(jax.jit, static_argnums=(1, 5))
def base_batch_sampler(key, batch_size, data, node_id, meta_data=None, num_devices=1):
    assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
    assert (
        node_id.ndim == 2 or node_id.ndim == 1
    ), "Node id must be 2D or 1D, (num_nodes, dim) or (num_nodes,)"

    index = jax.random.randint(key, shape=(num_devices,batch_size,), minval=0, maxval=data.shape[0])
    data_batch = data[index,...]
    node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
        jnp.int32
    )
    if meta_data is not None:
        if meta_data.ndim == 3:
            meta_data_batch = meta_data[index,...]
        else:
            meta_data_batch = jnp.repeat(meta_data[None, ...], num_devices, axis=0)
    else:
        meta_data_batch = None
    return data_batch, node_id_batch, meta_data_batch
    
    
    
    
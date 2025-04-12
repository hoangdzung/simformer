from sbibm import get_task as _get_torch_task

import torch
import jax
import jax.numpy as jnp
import numpy as np 

from scoresbibm.tasks.base_task import InferenceTask


class SBIBMTask(InferenceTask):
    observations = range(1, 11)

    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        self.task = _get_torch_task(self.name)
        
    def get_theta_dim(self):
        return self.task.dim_parameters
    
    def get_x_dim(self):
        return self.task.dim_data

    def get_prior(self):
        if self.backend == "torch":
            return self.task.get_prior_dist()
        else:
            raise NotImplementedError()

    def get_torch_prior(self):
        return self.task.get_prior_dist()
        
    def get_simulator(self):
        if self.backend == "torch":
            return self.task.get_simulator()
        else:
            raise NotImplementedError()
    
    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def get_data(self, num_samples: int, **kwargs):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return {"theta":thetas, "x":xs}
        except:
            print("aaaa")
            # If not implemented in JAX, use PyTorch
            old_backed = self.backend
            self.backend = "torch"
            prior = kwargs.get("proposal", None) or self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            self.backend = old_backed
            if self.backend == "numpy":
                thetas = thetas.numpy()
                xs = xs.numpy()
            elif self.backend == "jax":
                thetas = jnp.array(thetas)
                xs = jnp.array(xs)
            return {"theta":thetas, "x":xs}

    def get_observation(self, index: int):
        if self.backend == "torch":
            return self.task.get_observation(index)
        else:
            out = self.task.get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return self.task.get_reference_posterior_samples(index)
        else:
            out = self.task.get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int, x_o=None, num_samples=10000):
        if x_o is None:
            if self.backend == "torch":
                return self.task.get_reference_posterior_samples(index)
            else:
                out = self.task.get_reference_posterior_samples(index)
                if self.backend == "numpy":
                    return out.numpy()
                elif self.backend == "jax":
                    return jnp.array(out)
        else:    
            if self.backend == "torch":
                return self.task._sample_reference_posterior(num_samples=num_samples, observation=x_o)
            else:
                out = self.task._sample_reference_posterior(num_samples=num_samples, observation=torch.from_numpy(np.array(x_o)))
                if self.backend == "numpy":
                    return out.numpy()
                elif self.backend == "jax":
                    return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return self.task.get_true_parameters(index)
        else:
            out = self.task.get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)



class LinearGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_linear", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    
class LinearGaussianExtended(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_linear_extended", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
    
        base_mask = jnp.block([
            [thetas_mask, jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)],
            [jnp.zeros((x_dim, theta_dim), dtype=jnp.bool_), x_i_mask]
        ])
        
        
        # The last data (row) should only depend on itself
        base_mask = base_mask.at[-1, :].set(False)

        # The last data (row) should generate the other data (columns)
        base_mask = base_mask.at[-x_dim:, -1].set(True)
        
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    
class BernoulliGLM(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()
    

class BernoulliGLMRaw(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm_raw", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()
    



class MixtureGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_mixture", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        
class MixtureGaussianExtended(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_mixture_extended", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((x_dim, x_dim), dtype=jnp.bool_))
        
        base_mask = jnp.block([
            [thetas_mask, jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)],
            [jnp.ones((x_dim - 1, theta_dim), dtype=jnp.bool_), x_mask[:-1, :]],
            [jnp.zeros((1, theta_dim + x_dim), dtype=jnp.bool_)]
        ])
        base_mask = base_mask.astype(jnp.bool_)
        # The last data (row) should only depend on itself
        base_mask = base_mask.at[-1, -1].set(True)
        
        # The last data (row) should generate the other two data (column 2 and 3)
        base_mask = base_mask.at[2, -1].set(True)
        base_mask = base_mask.at[3, -1].set(True)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    

class TwoMoons(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="two_moons", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn

    def get_reference_posterior_samples(self, index: int, x_o=None, num_samples=10000):
        if x_o is None:
            if self.backend == "torch":
                return self.task.get_reference_posterior_samples(index)
            else:
                out = self.task.get_reference_posterior_samples(index)
                if self.backend == "numpy":
                    return out.numpy()
                elif self.backend == "jax":
                    return jnp.array(out)
        else:
    
            if self.backend == "torch":
                return self.task._sample_reference_posterior(num_samples=num_samples, num_observation=x_o.shape[0], observation=x_o)
            else:
                out = self.task._sample_reference_posterior(num_samples=num_samples, num_observation=x_o.shape[0], observation=torch.from_numpy(np.array(x_o)))    
                if self.backend == "numpy":
                    return out.numpy()
                elif self.backend == "jax":
                    return jnp.array(out)
class TwoMoonsExtended(SBIBMTask):
    observations = range(1, 2)
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="two_moons_extended", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((x_dim, x_dim), dtype=jnp.bool_))
        
        base_mask = jnp.block([
            [thetas_mask, jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)],
            [jnp.ones((x_dim - 1, theta_dim), dtype=jnp.bool_), x_mask[:-1, :]],
            [jnp.zeros((1, theta_dim + x_dim), dtype=jnp.bool_)]
        ])
        base_mask = base_mask.astype(jnp.bool_)
        # The last data (row) should only depend on itself
        base_mask = base_mask.at[-1, -1].set(True)
        
        # The last data (row) should generate the other two data (column 2 and 3)
        base_mask = base_mask.at[2, -1].set(True)
        base_mask = base_mask.at[3, -1].set(True)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    
class SLCP(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="slcp", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_) 
        # TODO This could be triangular -> DAG
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(*tuple([jnp.tril(jnp.ones((x_i_dim,x_i_dim), dtype=jnp.bool_))]*4)) 
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim,x_dim))], [jnp.ones((x_dim, theta_dim)), x_i_mask]]) 
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            # If node_ids are permuted, we need to permute the base_mask
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn

from typing import Optional, Dict, List
import torch
import pyro
import pyro.distributions as pdist
from pathlib import Path
import numpy as np
from scipy.integrate import ode
from scipy.optimize import fsolve
from sbibm.tasks.simulator import Simulator
import jax
import jax.numpy as jnp
from scoresbibm.tasks.base_task import CGMTask
from tqdm import tqdm 
from pathos.multiprocessing import ProcessingPool as Pool
import math 

MIN_CGM = 20
MAX_CGM = 600

def hovorka_model(t, x, u, D, P): ## This is the ode version
    """HOVORKA DIFFERENTIAL EQUATIONS
    # t:    Time window for the simulation. Format: [t0 t1], or [t1 t2 t3 ... tn]. [min]
    # x:    Initial conditions
    # u:    Amount of insulin insulin injected [mU/min]
    # D:    CHO eating rate [mmol/min]
    # P:    Model fixed parameters
    #
    # Syntax :
    # [T, X] = ode15s(@Hovorka, [t0 t1], xInitial0, odeOptions, u, D, p);
    """
    # TODO: update syntax in docstring

    # u, D, P = args

    # Defining the various equation names
    D1 = x[ 0 ]               # Amount of glucose in compartment 1 [mmol]
    D2 = x[ 1 ]               # Amount of glucose in compartment 2 [mmol]
    S1 = x[ 2 ]               # Amount of insulin in compartment 1 [mU]
    S2 = x[ 3 ]               # Amount of insulin in compartment 2 [mU]
    Q1 = x[ 4 ]               # Amount of glucose in the main blood stream [mmol]
    Q2 = x[ 5 ]               # Amount of glucose in peripheral tissues [mmol]
    I =  x[ 6 ]                # Plasma insulin concentration [mU/L]
    x1 = x[ 7 ]               # Insluin in muscle tissues [1], x1*Q1 = Insulin dependent uptake of glucose in muscles
    x2 = x[ 8 ]               # [1], x2*Q2 = Insulin dependent disposal of glucose in the muscle cells
    x3 = x[ 9 ]              # Insulin in the liver [1], EGP_0*(1-x3) = Endogenous release of glucose by the liver
    C = x[10]

    # Unpack data
    tau_G = P[ 0 ]               # Time-to-glucose absorption [min]
    tau_I = P[ 1 ]               # Time-to-insulin absorption [min]
    A_G = P[ 2 ]                 # Factor describing utilization of CHO to glucose [1]
    k_12 = P[ 3 ]                # [1/min] k_12*Q2 = Transfer of glucose from peripheral tissues (ex. muscle to the blood)
    k_a1 = P[ 4 ]                # Deactivation rate [1/min]
    k_b1 = P[ 5 ]                # [L/(mU*min)]
    k_a2 = P[ 6 ]                # Deactivation rate [1/min]
    k_b2 = P[ 7 ]                # [L/(mU*min)]
    k_a3 = P[ 8 ]                # Deactivation rate [1/min]
    k_b3 = P[ 9 ]               # [L/(mU*min)]
    k_e = P[ 10 ]                # Insulin elimination rate [1/min]
    V_I = P[ 11 ]                # Insulin distribution volume [L]
    V_G = P[ 12 ]                # Glucose distribution volume [L]
    F_01 = P[ 13 ]               # Glucose consumption by the central nervous system [mmol/min]
    EGP_0 = P[ 14 ]              # Liver glucose production rate [mmol/min]

    # If some parameters are not defined
    if len(P) == 15:
        ka_int = 0.073
        R_cl = 0.003
        # R_thr = 9
        R_thr = 14
    elif len(P) == 18:
        R_cl = P[16]
        ka_int = P[15]
        R_thr = P[17]

    # Certain parameters are defined
    U_G = D2/tau_G             # Glucose absorption rate [mmol/min]
    U_I = S2/tau_I             # Insulin absorption rate [mU/min]

    # Constitutive equations
    G = Q1/V_G                 # Glucose concentration [mmol/L]

    # if (G>=4.5):
    #     F_01c = F_01           # Consumption of glucose by the central nervous system [mmol/min
    # else:
    #     F_01c = F_01*G/4.5     # Consumption of glucose by the central nervous system [mmol/min]

    F_01s = F_01/0.85
    F_01c = F_01s*G / (G + 1)

    # if (G>=9):
        # F_R = 0.003*(G-9)*V_G  # Renal excretion of glucose in the kidneys [mmol/min]
    # else:
        # F_R = 0                # Renal excretion of glucose in the kidneys [mmol/min]

    if (G >= R_thr):
        F_R = R_cl*(G - R_thr)*V_G  # Renal excretion of glucose in the kidneys [mmol/min]
    else:
        F_R = 0                # Renal excretion of glucose in the kidneys [mmol/min]

    # Mass balances/differential equations
    xdot = np.zeros (11);

    xdot[ 0 ] = A_G*D-D1/tau_G                                # dD1
    xdot[ 1 ] = D1/tau_G-U_G                                  # dD2
   
    xdot[ 2 ] = u-S1/tau_I                                    # dS1
    xdot[ 3 ] = S1/tau_I-U_I                                  # dS2
   
    xdot[ 4 ] = -(F_01c+F_R) - x1*Q1 + k_12*Q2 + U_G + max(EGP_0*(1-x3), 0)   # dQ1
    xdot[ 5 ] = x1*Q1-(k_12+x2)*Q2                            # dQ2

    xdot[ 6 ] = U_I/V_I-k_e*I                                 # dI
    
    xdot[ 7 ] = k_b1*I-k_a1*x1                                # dx1
    xdot[ 8 ] = k_b2*I-k_a2*x2                                # dx2
    xdot[ 9 ] = k_b3*I-k_a3*x3                                # dx3

    # ===============
    # CGM delay
    # ===============
    xdot[10] = ka_int*(G - C)


    return xdot


def hovorka_model_tuple(x, *pars):
    """HOVORKA DIFFERENTIAL EQUATIONS without time variable
    # t:    Time window for the simulation. Format: [t0 t1], or [t1 t2 t3 ... tn]. [min]
    # x:    Initial conditions
    # u:    Amount of insulin insulin injected [mU/min]
    # D:    CHO eating rate [mmol/min]
    # P:    Model fixed parameters
    #
    """
    # TODO: update syntax in docstring
    # import numpy as np

    u, D, P = pars

    t = 0

    xdot = hovorka_model(t, x, u, D, P)

    return xdot


def run_simulation(P, meal_vector, X=5):  # Default sampling every 5 minutes
    
    # Initial values for parameters
    init_basal = 6.43
    initial_pars = (init_basal, 0, P)

    # Bolus carb factor -- [g/U]
    carb_factor = 25

    # Initial value
    # X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=initial_pars)
    # print(X0)
    
    X0 = np.array([-1.18193253e-23, -2.36386506e-23,  3.53650000e+02,  3.53650000e+02,
                6.54454634e+01,  2.63459848e+01,  5.54692892e+00,  2.84002761e-02,
                4.54848171e-03,  2.88440304e-01,  5.84334494e+00])

    # Simulation setup
    integrator = ode(hovorka_model)
    # integrator.set_integrator('vode', method='bdf', order=4)
    integrator.set_integrator('dopri5')
    integrator.set_initial_value(X0, 0)

    action = init_basal

    blood_glucose_value = []
            
    for i in range(1440):  # Step every X minutes

        if meal_vector[i] > 0:
            insulin_rate = action + np.round(max(meal_vector[i] * (180 / carb_factor), 0), 1)
        else:
            insulin_rate = action

        # Updating the carb and insulin parameters in the model
        integrator.set_f_params(insulin_rate, meal_vector[i], P)

        # Integration step
        integrator.integrate(integrator.t + X)  # Advance by X minutes

        # blood_glucose_value.append(int(integrator.y[-1]*18))
        value = integrator.y[-1] * 18

        # Clamp +inf to 600, -inf to 20
        if math.isinf(value):
            value = 600 if value > 0 else 20

        blood_glucose_value.append(int(value))

    # Returning blood glucose value
    return np.array(blood_glucose_value[::X])


class HovorkaBaseTask(CGMTask):
            
    test_params = torch.tensor([
            [0.0343, 0.0031, 0.0752, 0.0472, 29.4e-4, 0.9e-4, 401e-4, 0.18, 0.0121, 0.0148, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
            [0.0871, 0.0157, 0.0231, 0.0143, 18.7e-4, 6.1e-4, 379e-4, 0.13, 0.0075, 0.0143, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
            [0.0863, 0.0029, 0.0495, 0.0691, 81.2e-4, 20.1e-4, 578e-4, 0.22, 0.0103, 0.0156, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
            [0.0968, 0.0088, 0.0302, 0.0118, 86.1e-4, 4.7e-4, 720e-4, 0.14, 0.0119, 0.0213, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
            [0.0390, 0.0007, 0.1631, 0.0114, 72.4e-4, 15.3e-4, 961e-4, 0.14, 0.0071, 0.02, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
            [0.0458, 0.0017, 0.0689, 0.0285, 19.1e-4, 2.2e-4, 82e-4, 0.13, 0.0092, 0.0105, 40, 80, 60, 30, 8*60, 12*60, 18*60, 22*60, 70],
        ]).float()
    num_tests = test_params.shape[0]
    
    def __init__(self, backend: str = "jax", T: int = 10) -> None:
        """Hovorka Model Task"""
        super().__init__("Hovorka Base", backend)
        self.num_simulations = [100, 1000, 10000, 100000, 1000000]
        self.T = T
        self.dim_external = 0
        self.dim_data =  1440 // self.T

        # Define prior distributions for uncertain parameters
        self.constant_params = {
            "V_I" : 0.12,
            "tau_G" : 40,
            "tau_I" : 55,
            "A_G" : 0.8,
            "k_e" : 0.138,
        }
        self.prior_params = {
            "k_12": {"mean": 0.0649, "std": 0.0115},
            "k_a1": {"mean": 0.0055, "std": 0.0023},
            "k_a2": {"mean": 0.0683, "std": 0.0207},
            "k_a3": {"mean": 0.0304, "std": 0.0096},
            "S_IT": {"mean": 51.2e-4, "std": 13.1e-4},
            "S_ID": {"mean": 8.2e-4, "std": 3.2e-4},
            "S_IE": {"mean": 520e-4, "std": 125e-4},
            "V_G": {"mean": 0.16, "std": 0.01},
            "F_01": {"mean": 0.0097, "std": 0.0009},
            "EGP_0": {"mean": 0.0161, "std": 0.0001},
        }

        self.dim_parameters = len(self.prior_params)
        # Define a Pyro distribution for each parameter
        self.prior_dist = pdist.Independent(
            pdist.Normal(
                torch.tensor([self.prior_params[param]["mean"] for param in self.prior_params]),
                torch.tensor([self.prior_params[param]["std"] for param in self.prior_params])
            ),
            1
        )
        self.prior_dist.set_default_validate_args(False)

    def _sample_meals(self, num_samples):
        """Sample meal sizes and times (integers)."""
        base_meals = np.array([40, 80, 60, 30])
        base_times = np.array([8*60, 12*60, 18*60, 22*60])
    
        meals = np.clip(
            np.random.normal(loc=base_meals, scale=15, size=(num_samples, 4)),
            a_min=10, a_max=120
        )
        meal_times = np.clip(
            np.random.normal(loc=base_times, scale=30, size=(num_samples, 4)),
            a_min=6*60, a_max=24*60 - 1
        )
    
        return np.round(meals).astype(int), np.round(meal_times).astype(int)
    
    def _sample_BWs(self, num_samples):
        """Sample body weights from a uniform distribution."""
        return np.random.uniform(40, 120, size=(num_samples,))

    def _prepare_simulation_params(self, parameters, BWs, meals_data):
        num_samples = parameters.shape[0]
        
        meals, meal_times = meals_data
        if not isinstance(meals, np.ndarray):
            meals, meal_times = np.round(meals.numpy()).astype(int), np.round(meal_times.numpy()).astype(int)
        if not isinstance(BWs, np.ndarray):
            BWs = BWs.numpy()
        # Prepare parameters efficiently
        params = np.column_stack([
            np.full(num_samples, self.constant_params["tau_G"]),
            np.full(num_samples, self.constant_params["tau_I"]),
            np.full(num_samples, self.constant_params["A_G"]),
            parameters[:, 0],  # k_12
            parameters[:, 1],  # k_a1
            parameters[:, 1] * parameters[:, 4],  # k_b1 = k_a1 * S_IT
            parameters[:, 2],  # k_a2
            parameters[:, 2] * parameters[:, 5],  # k_b2 = k_a2 * S_ID
            parameters[:, 3],  # k_a3
            parameters[:, 3] * parameters[:, 6],  # k_b3 = k_a3 * S_IE
            np.full(num_samples, self.constant_params["k_e"]),
            np.full(num_samples, self.constant_params["V_I"]) * BWs,
            parameters[:, 7] * BWs,  # V_G
            parameters[:, 8] * BWs,  # F_01
            parameters[:, 9] * BWs  # EGP_0
        ])
        assert params.shape[1] == 15, params.shape
        meal_vectors = np.zeros((num_samples, 1440))
        meal_vectors[np.arange(num_samples)[:, None], meal_times] = meals * 1000/180
        
        return params, meal_vectors 

    def get_test_params(self, index) -> torch.Tensor:
        return self.test_params[index]

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """
        X = self.T
        def _run_simulation_wrapper(args):
            params, meal_vector, X = args
            out = run_simulation(params, meal_vector, X=X)
            out = np.clip(out.astype(float), a_min=MIN_CGM, a_max=MAX_CGM)
            return torch.tensor(out)
        
        def simulator(parameters):
            num_samples, num_parameters = parameters.shape

            if num_parameters == self.dim_parameters + 9:
                meals_data = parameters[:, -9:-5], parameters[:, -5:-1]
                BWs = parameters[:, -1]
            elif num_parameters == self.dim_parameters + 1:
                BWs = parameters[:, -1]
                meals_data = self._sample_meals(num_samples)
            elif num_parameters == self.dim_parameters + 8:
                meals_data = parameters[:, -8:-4], parameters[:, -4:]
                BWs = self._sample_BWs(num_samples)
            elif num_parameters == self.dim_parameters:
                meals_data = self._sample_meals(num_samples)
                BWs = self._sample_BWs(num_samples)

            params, meal_vectors = self._prepare_simulation_params(parameters, BWs, meals_data)

            # Use multiprocessing to speed up the simulation
            with Pool() as pool:  # Uses pathos instead of multiprocessing
                results = list(tqdm(pool.imap(_run_simulation_wrapper, zip(params, meal_vectors, [X] * num_samples)), 
                                    total=num_samples, desc="Simulating"))
            results = torch.stack(results)

            return results.float()
        return Simulator(task=self, simulator=simulator, max_calls=max_calls, external_inp=True)

    def get_data(self, num_samples: int, **kwargs):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return {"theta":thetas, "x":xs }
        except:
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
                thetas = jax.device_put(jnp.array(thetas), jax.devices("cpu")[0])
                xs = jax.device_put(jnp.array(xs), jax.devices("cpu")[0])

            return {"theta":thetas, "x":xs}

    def get_base_mask_fn(self):
        theta_dim = self.dim_parameters
        x_dim = self.dim_data

        # Identity matrix for parameters (each param depends on itself)
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)

        # Parameter-to-data connections (params generate all data, full of 1s)
        param_to_data = jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)

        # Data-to-param connections (should be zeros)
        data_to_param = jnp.ones((x_dim, theta_dim), dtype=jnp.bool_)

        # Each data point depends only on itself
        x_mask = jnp.eye(x_dim, dtype=jnp.bool_)

        # Previous data generates the next data (superdiagonal ones)
        x_mask = x_mask.at[1:, :-1].set(jnp.eye(x_dim - 1, dtype=jnp.bool_)) | x_mask
 
        # Combine all parts into a full mask
        base_mask = jnp.block([
            [thetas_mask, param_to_data],  # Params depend on themselves, generate all data
            [data_to_param, x_mask]        # Data depends on itself, prev data generates next
        ])
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        
        
class HovorkaTask(HovorkaBaseTask):
            
    def __init__(self, backend: str = "jax", T: int = 10) -> None:
        """Hovorka Model Task"""

        super().__init__(backend, T)
        self.name = "Hovorka"
        self.dim_external = 9
        self.dim_data += self.dim_external
    
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """
        X = self.T
        def _run_simulation_wrapper(args):
            params, meal_vector, X = args
            out = run_simulation(params, meal_vector, X=X)
            out = np.clip(out.astype(float), a_min=MIN_CGM, a_max=MAX_CGM)
            return torch.tensor(out)
        
        def simulator(parameters):
            num_samples, num_parameters = parameters.shape

            if num_parameters == self.dim_parameters + 9:
                meals_data = parameters[:, -9:-5], parameters[:, -5:-1]
                BWs = parameters[:, -1]
            elif num_parameters == self.dim_parameters + 1:
                BWs = parameters[:, -1]
                meals_data = self._sample_meals(num_samples)
            elif num_parameters == self.dim_parameters + 8:
                meals_data = parameters[:, -8:-4], parameters[:, -4:]
                BWs = self._sample_BWs(num_samples)
            elif num_parameters == self.dim_parameters:
                meals_data = self._sample_meals(num_samples)
                BWs = self._sample_BWs(num_samples)

            params, meal_vectors = self._prepare_simulation_params(parameters, BWs, meals_data)

            # Use multiprocessing to speed up the simulation
            with Pool() as pool:  # Uses pathos instead of multiprocessing
                results = list(tqdm(pool.imap(_run_simulation_wrapper, zip(params, meal_vectors, [X] * num_samples)), 
                                    total=num_samples, desc="Simulating"))
            results = torch.stack(results)
            # Concatenate meal_data and BWs to results
            meal_data_concat = torch.cat([torch.tensor(meals_data[0]), torch.tensor(meals_data[1])], dim=1)  # Merge meal_data parts
            BWs_tensor = torch.tensor(BWs)[:, None]
            final_output = torch.cat([results, meal_data_concat, BWs_tensor], dim=1)  # Concatenate everything

            return final_output.float()
        return Simulator(task=self, simulator=simulator, max_calls=max_calls, external_inp=True)
    
    def get_data(self, num_samples: int, **kwargs):
        data = super().get_data(num_samples, **kwargs)  # Call parent method
        
        # # Concatenate theta and x based on the backend
        # if self.backend == "jax":
        #     metadata = jnp.concatenate([data["theta"], data["x"]], axis=-1)
        # elif self.backend == "torch":
        #     metadata = torch.cat([data["theta"], data["x"]], dim=-1)
        # elif self.backend == "numpy":
        #     metadata = np.concatenate([data["theta"], data["x"]], axis=-1)
        # else:
        #     raise ValueError(f"Unsupported backend: {self.backend}")
        theta, x = data["theta"], data["x"]

        if self.backend == "jax":
            import jax.numpy as jnp
            nan = jnp.nan
            ones = jnp.ones_like
            linspace = jnp.linspace
            concat = jnp.concatenate

            N = x.shape[0]
            D = x.shape[1]
            P = theta.shape[1]

            metadata = nan * ones(concat([theta, x], axis=-1))

            # Set -5:-1 of metadata to be the same as -5:-1 of x
            metadata = metadata.at[:, P + D - 5:P + D - 1].set(x[:, -5:-1])

            # Set 0:-9 of metadata to time index
            time_index = linspace(0, 1439, D - 9)  / 1.0
            time_index = jnp.tile(time_index[None, :], (N, 1))
            metadata = metadata.at[:, P:P + D - 9].set(time_index)

        elif self.backend == "torch":
            import torch
            nan = torch.nan
            ones = torch.ones_like
            linspace = lambda start, end, steps: torch.linspace(start, end, steps, device=x.device)
            concat = torch.cat

            N = x.shape[0]
            D = x.shape[1]
            P = theta.shape[1]

            metadata = nan * ones(concat([theta, x], dim=-1))

            metadata[:, P + D - 5:P + D - 1] = x[:, -5:-1]

            time_index = linspace(0, 1439, D - 9) / 1.0
            time_index = time_index.unsqueeze(0).repeat(N, 1)
            metadata[:, P:P + D - 9] = time_index

        elif self.backend == "numpy":
            import numpy as np
            nan = np.nan
            ones = np.ones_like
            linspace = np.linspace
            concat = np.concatenate

            N = x.shape[0]
            D = x.shape[1]
            P = theta.shape[1]

            metadata = nan * ones(concat([theta, x], axis=-1))

            metadata[:, P + D - 5:P + D - 1] = x[:, -5:-1]

            time_index = linspace(0, 1439, D - 9) / 1.0
            time_index = np.tile(time_index[None, :], (N, 1))
            metadata[:, P:P + D - 9] = time_index

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


        # Add metadata to the returned dictionary
        data["metadata"] = metadata
        return data

    def get_base_mask_fn(self):
        theta_dim = self.dim_parameters
        x_dim = self.dim_data - self.dim_external
        ext_dim = self.dim_external

        base_mask = jnp.eye(theta_dim + x_dim + ext_dim, dtype=jnp.bool_)

        base_mask = base_mask.at[theta_dim: theta_dim + x_dim, :theta_dim].set(True)
        base_mask = base_mask.at[theta_dim: theta_dim + x_dim:, -1].set(True)

        # Previous data generates the next data (superdiagonal ones)
        base_mask = base_mask.at[theta_dim + 1: theta_dim + x_dim, theta_dim: theta_dim + x_dim - 1].set(jnp.eye(x_dim - 1, dtype=jnp.bool_)) | base_mask

        X = self.T
        def base_mask_fn(node_ids, node_meta_data):
            node_meta_data = node_meta_data.reshape(node_ids.shape)
            mask = base_mask[node_ids, :][:, node_ids]
            
            meal_time = node_meta_data[-5:-1].astype(int)
            meal_time_subsampled = jnp.ceil(meal_time / X).astype(int)
            # Generate row and column indices for setting values
            row_indices = meal_time_subsampled.repeat(2) + theta_dim 
            col_indices = jnp.array([-9, -5, -8, -4, -7, -3, -6, -2]) 

            # Set the corresponding entries to True
            return mask.at[row_indices, col_indices].set(True)
            
        return base_mask_fn
        
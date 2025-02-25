
import sbi 
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils import RestrictedPrior
from scoresbibm.methods.models import SBIPosteriorModel

from scoresbibm.methods.score_sbi import train_conditional_score_model
from scoresbibm.methods.score_transformer import train_transformer_model
from scoresbibm.utils.restriction_estimator import get_jax_density_thresholder


def run_npe_default(task,data, method_cfg, rng=None):
    """ Train a default SBI model"""
    device = method_cfg.device
    thetas, xs = data["theta"], data["x"]
    density_estimator = posterior_nn(**method_cfg.model)
    inference = SNPE(density_estimator=density_estimator, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.train)
    
    # Output is sampling_fn
    posterior = inference.build_posterior(**method_cfg.posterior)
    
    model = SBIPosteriorModel(posterior, method="npe")
    return model


def run_nle_default(task, data, method_cfg, rng=None):
    device = method_cfg.device
    thetas, xs = data["theta"], data["x"]
    density_estimator = likelihood_nn(**method_cfg.model)
    inference = SNLE(prior = task.get_prior(),density_estimator=density_estimator, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.train)
    
    posterior = inference.build_posterior(**method_cfg.posterior)
    model = SBIPosteriorModel(posterior, method="nle")
    return model


def run_nre_default(task, data, method_cfg, rng=None):
    device = method_cfg.device
    thetas, xs = data["theta"], data["x"]
    classifier = classifier_nn(**method_cfg.model)
    inference = SNRE(prior = task.get_prior(), classifier=classifier, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.train)
    
    posterior = inference.build_posterior(**method_cfg.posterior)
    model = SBIPosteriorModel(posterior, method="nre")
    return model

def run_nspe(task, data, method_cfg, rng=None):
    thetas, xs = data["theta"], data["x"]
    model = train_conditional_score_model(task, thetas, xs, method_cfg, rng)
    model.set_default_sampling_kwargs(**method_cfg.posterior)
    return model

def run_score_transformer(task, data, method_cfg, rng=None):
    model = train_transformer_model(task, data, method_cfg, rng)
    model.set_default_sampling_kwargs(**method_cfg.posterior)
    return model

def run_tp_score_transformer(task, data, method_cfg, rng=None, x_o=None, num_observations=1, n_rounds=5):
    if x_o is None:
        x_o = task.task.get_observation(num_observations)
    prev_params = None
    num_sims = data['x'].shape[0]
    prior = task.get_torch_prior()
    for _ in range(n_rounds):
        model = train_transformer_model(task, data, method_cfg, rng, prev_params=prev_params)
        model.set_default_x_o(x_o)
        accept_reject_fn = get_jax_density_thresholder(model, rng=rng, quantile=1e-4, num_samples_to_estimate_support=10_000)
        proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
        data = task.get_data(num_sims, rng=rng, proposal=proposal)
        prev_params = model.params

    model.set_default_sampling_kwargs(**method_cfg.posterior)
    return model


def get_method(name:str):
    """ Get a method"""
    if name == "npe":
        return run_npe_default
    elif name == "nle":
        return run_nle_default
    elif name == "nre":
        return run_nre_default
    elif name == "nspe":
        return run_nspe
    elif "tp_score_transformer" in name:
        return run_tp_score_transformer
    elif "score_transformer" in name:
        return run_score_transformer
    else:
        raise NotImplementedError()
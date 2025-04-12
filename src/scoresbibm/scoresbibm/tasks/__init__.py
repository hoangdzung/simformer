

from scoresbibm.tasks.sbibm_tasks import LinearGaussian, MixtureGaussian, TwoMoons, SLCP, BernoulliGLM, BernoulliGLMRaw, TwoMoonsExtended, LinearGaussianExtended, MixtureGaussianExtended
from scoresbibm.tasks.all_conditional_tasks import TwoMoonsAllConditionalTask, SLCPAllConditionalTask, NonlinearGaussianTreeAllConditionalTask, NonlinearMarcovChainAllConditionalTask
from scoresbibm.tasks.unstructured_tasks import LotkaVolterraTask, SIRTask
from scoresbibm.tasks.hhtask import HHTask
from scoresbibm.tasks.cgm_tasks import HovorkaTask, HovorkaBaseTask

def get_task(name: str, backend: str = "jax"):
    if name == "gaussian_linear":
        return LinearGaussian(backend=backend)
    elif name == "gaussian_linear_extended":
        return LinearGaussianExtended(backend=backend)
    elif name == "gaussian_mixture":
        return MixtureGaussian(backend=backend)
    elif name == "gaussian_mixture_extended":
        return MixtureGaussianExtended(backend=backend)
    elif name == "two_moons":
        return TwoMoons(backend=backend)
    elif name == "two_moons_extended":
        return TwoMoonsExtended(backend=backend)
    elif name == "slcp":
        return SLCP(backend=backend)
    elif name == "bernoulli_glm":
        return BernoulliGLM(backend=backend)
    elif name == "bernoulli_glm_raw":
        return BernoulliGLMRaw(backend=backend)
    elif name == "two_moons_all_cond":
        return TwoMoonsAllConditionalTask(backend=backend)
    elif name == "slcp_all_cond":
        return SLCPAllConditionalTask(backend=backend)
    elif name == "tree_all_cond":
        return NonlinearGaussianTreeAllConditionalTask(backend=backend)
    elif name == "marcov_chain_all_cond":
        return NonlinearMarcovChainAllConditionalTask(backend=backend)
    elif name == "lotka_volterra":
        return LotkaVolterraTask(backend=backend)
    elif name == "sir":
        return SIRTask(backend=backend)
    elif name == "hh":
        return HHTask(backend=backend)
    elif name == "hovorka":
        return HovorkaTask(backend=backend)
    elif name == "hovorka_base":
        return HovorkaBaseTask(backend=backend)
    else:
        raise NotImplementedError()
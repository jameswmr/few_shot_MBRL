import random
import numpy as np
import torch
from gym_env.utilities import ENV_CONTEXTS, SAC_CONFIG, PPO_CONFIG
from gym_env.utilities import RandomContextStateWrapper, NormalizeActionSpaceWrapper
from stable_baselines3.common.monitor import Monitor
import os

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
def make_env_f(env_name, dr):
    def _make_env():
        Env = ENV_CONTEXTS[env_name]["constructor"]
        env = Env(render=False)
        env = RandomContextStateWrapper(env, env_name, dr)
        env = NormalizeActionSpaceWrapper(env)
        return env
    # env = Monitor(env)
    return _make_env

def make_env(env_name, dr):
    Env = ENV_CONTEXTS[env_name]['constructor']
    env = Env(render=False)
    env = RandomContextStateWrapper(env, env_name, dr)
    env = NormalizeActionSpaceWrapper(env)
    return env

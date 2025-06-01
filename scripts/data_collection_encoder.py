import os
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from gym_env.utilities import ENV_CONTEXTS, SAC_CONFIG, PPO_CONFIG
from gym_env.utilities import RandomContextStateWrapper, NormalizeActionSpaceWrapper
from scripts.utilities import set_seed, make_env, parse_argument
from torch.utils.data import TensorDataset, DataLoader
from gym.vector import AsyncVectorEnv
from scripts.train_dynamic import DynamicModel, RNNModel


def main(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)
    # dynamic = DynamicModel()
    dynamic = RNNModel(device=args.device)
    dynamic.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    # env = AsyncVectorEnv([make_env(args.env_name, args.dr) for _ in range(args.n_envs)])
    env = make_env(args.env_name, args.dr)
    # print(env)
    trajectory = []
    real_trajectory = []
    obss = []
    actions = []
    env_params = []
    for _ in tqdm(range(args.n_episodes)):
        obs = env.reset()
        env_params.append(obs.copy()[2:])
        obs[2:] = [np.random.uniform(-1, 1) for _ in range(len(obs)-2)]
        obss.append(obs)
        action = env.action_space.sample()
        actions.append(action)
        obs = np.repeat(obs[None, :], 40, axis=0)
        action = np.repeat(action[None, :], 40, axis=0)
        # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(args.n_envs)]), device=args.device)
        with torch.no_grad():
            dynamic.eval()
            trajectories = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu().squeeze(0)
        trajectory.append(trajectories)
        nxt_obs, reward, done, info = env.step(action[0])
        real_trajectory.append(env.get_state_traj(info))
    trajectory_tensor = torch.stack(trajectory)
    real_trajectory_tensor = torch.tensor(np.array(real_trajectory), dtype=torch.float32)

    obss_tensor = torch.tensor(np.array(obss), dtype=torch.float32)      
    actionss_tensor = torch.tensor(np.array(actions), dtype=torch.float32) 
    env_params_tensor = torch.tensor(np.array(env_params), dtype=torch.float32)
    data = {
        'trajectory': trajectory_tensor,
        'real_trajectory': real_trajectory_tensor,
        'obs': obss_tensor, 
        'actions': actionss_tensor,
        'env_params': env_params_tensor,
    }
    torch.save(data, args.output_dir)





    

if __name__ == "__main__":
    args = parse_argument()
    main(args)
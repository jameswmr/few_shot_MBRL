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
from scripts.utilities import set_seed, make_env
from torch.utils.data import TensorDataset, DataLoader
from gym.vector import AsyncVectorEnv
from scripts.train_dynamic import DynamicModel


def main(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)
    dynamic = DynamicModel()
    dynamic.model.load_state_dict(torch.load(f'output/{args.model_dir}'))
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
        obs[2:] = [0 for _ in range(len(obs)-2)]
        obss.append(obs)
        action = env.action_space.sample()
        actions.append(action)
        # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(args.n_envs)]), device=args.device)
        trajectories = []
        for t in range(40):
            with torch.no_grad():
                trajectories.extend(np.array(dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu()))
        trajectory.append(trajectories)
        nxt_obs, reward, done, info = env.step(action)
        real_trajectory.append(env.get_state_traj(info))
    trajectory_tensor = torch.tensor(np.array(trajectory), dtype=torch.float32)
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
    # Define available environment names from the context
    env_names = list(ENV_CONTEXTS.keys())

    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, choices=env_names, help="Environment name"
    )
    parser.add_argument(
        "--policy_name", type=str, choices=["ppo", "sac"], help="Policy name"
    )
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=150000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--save_freq", type=int, default=150, help="Checkpoint interval"
    )
    parser.add_argument(
        "--pre_trained_model",type=str, default=None, help="Pre-trained model path"
    )
    
    parser.add_argument(
        "--dr", action="store_true", default=False, help="Enable domain randomization"
    )
    
    parser.add_argument(
        "--n_episodes", type=int, default=10000, help="Data Collection for n episodes"
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Set device"
    )

    parser.add_argument(
        "--output_dir", type=str, default="data/data.pt", help="Set output path"
    )

    parser.add_argument(
        "--final_state", type=bool, default=False, help="Only collect final state"
    )

    parser.add_argument(
        "--model_dir", type=str, default="best_dynamic_model.pt", help="Set load model directory"
    )
    # Parse arguments and call the main function
    args = parser.parse_args()
    main(args)
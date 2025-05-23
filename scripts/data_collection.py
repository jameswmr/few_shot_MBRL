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


def main(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)

    # env = AsyncVectorEnv([make_env(args.env_name, args.dr) for _ in range(args.n_envs)])
    env = make_env(args.env_name, args.dr)
    # print(env)
    obss = []
    actionss = []
    statess = []
    times = []
    for _ in tqdm(range(args.n_episodes)):
        obs = env.reset()
        # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(args.n_envs)]), device=args.device)
        actions = env.action_space.sample()
        nxt_obs, reward, done, info = env.step(actions)
        trajectories = env.get_state_traj(info)
        # print(trajectories)
        if args.final_state:
            obss.append(obs)
            actionss.append(actions)
            times.append(39)
            statess.append(trajectories[-1])
        else: 
            for t in range(40):
                obss.append(obs)
                actionss.append(actions)
                times.append(t)
                statess.append(trajectories[t])
    obss_tensor = torch.tensor(np.array(obss), dtype=torch.float32)      
    actionss_tensor = torch.tensor(np.array(actionss), dtype=torch.float32) 
    times_tensor = torch.tensor(np.array(times), dtype=torch.float32).unsqueeze(1)  
    statess_tensor = torch.tensor(np.array(statess), dtype=torch.float32) 
    data = {
        'obs': obss_tensor,
        'actions': actionss_tensor,
        'times': times_tensor,
        'states': statess_tensor
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
    # Parse arguments and call the main function
    args = parser.parse_args()
    main(args)
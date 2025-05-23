from scripts.train_dynamic import DynamicModel
from scripts.train_encoder import Encoder
from scripts.utilities import set_seed, make_env
from tqdm import tqdm
from gym_env.utilities import ENV_CONTEXTS
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
def main(args):
    dynamic = DynamicModel()
    dynamic.model.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    encoder = Encoder()
    encoder.model.load_state_dict(torch.load(f'output/{args.enc_dir}'))
    set_seed(args.seed)

    env = make_env(args.env_name, args.dr)
    #torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([t], device=args.device).unsqueeze(0)).cpu()

    e_dr_real = env.get_context()

    e_dr_real["dynamic@floor1_table_collision@friction_sliding"] = 0.2
    e_dr_real["dynamic@floor2_table_collision@friction_sliding"] = 0.2
    e_dr_real["dynamic@knob_g0@damping_ratio"] = -3
    e_dr_real["dynamic@x_left_wall_g0@damping_ratio"] = -3
    e_dr_real["dynamic@x_right_wall_g0@damping_ratio"] = -3
    e_dr_real["dynamic@y_front_wall_g0@damping_ratio"] = -3
    guess_env_params = [0 for _ in range(6)]

    for _ in range(10):
        env.set_context(e_dr_real)
        env.env.defined_puck_pos = [-0.147, 0.036]
        obs = env.reset()
        obs[2:] = guess_env_params
        best_action = None
        best_reward = float('-inf')
        for _ in range(1000): #Parallel?
            action = env.action_space.sample()
            with torch.no_grad():
                final_state = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu()
            reward = -np.linalg.norm(np.array(final_state) - np.array(env.target_pos[:2]))
            if reward > best_reward:
                best_action = action
                best_reward = reward
        nxt_obs, reward, done, info = env.step(best_action)
        print(reward)
        trajectories = env.get_state_traj(info)

        with torch.no_grad():
            guess_env_params = encoder.forward(torch.tensor(trajectories, device=args.device).unsqueeze(0), torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0)).cpu()
        



if __name__ == "__main__":
    # Define available environment names from the context
    env_names = list(ENV_CONTEXTS.keys())

    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, choices=env_names, help="Environment name"
    )
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    parser.add_argument(
        "--dr", type=bool , default=False, help="Enable domain randomization"
    )
    
    parser.add_argument(
        "--n_episodes", type=int, default=100, help="Data Collection for n episodes"
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Set device"
    )

    parser.add_argument(
        "--dir", type=str, default='best_dynamic_model.pt', help="Set load model directory"
    )
    # Parse arguments and call the main function
    args = parser.parse_args()
    main(args)   
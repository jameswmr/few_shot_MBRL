from baseline2_stage_2 import AdaptationModule, get_traj
from baseline2_stage_1 import make_env
import argparse
import torch
from copy import deepcopy
from stable_baselines3 import PPO
import numpy as np
from tqdm import tqdm
import random
import os
from random import uniform

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--am_model", type=str, required=True)
    parser.add_argument("--base_policy_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ood", action="store_true")
    args = parser.parse_args()

    # Set all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    adaptation_module = AdaptationModule()
    adaptation_module.load(args.am_model)
    base_policy = PPO.load(args.base_policy_model)
    
    # Set the seed for the base policy
    base_policy.set_random_seed(args.seed)
    
    headless = True
    state_dim = 2
    priv_dim = 9

    env = make_env("push_one", args.seed, headless, state_dim, priv_dim)()

    all_rewards = []
    for _ in tqdm(range(20)):

        obs = env.reset()

        # Modify friction for OOD evaluation
        if args.ood:
            e_dr_real = env.unwrapped.env.get_context()
            e_dr_real["dynamic@floor1_table_collision@friction_sliding"] = uniform(0.02, 0.2)
            e_dr_real["dynamic@floor2_table_collision@friction_sliding"] = uniform(0.02, 0.2)
            env.unwrapped.env.set_context(e_dr_real)

        # Estimate the priv info
        sim_env = deepcopy(env)
        cup_pos = torch.from_numpy(obs["obs"]).float()
        action = torch.from_numpy(sim_env.action_space.sample()).float()
        obs, reward, done, info = sim_env.step(action)
        cup_traj = torch.from_numpy(get_traj(info)).flatten().float()
        am_inputs = torch.cat([cup_pos, action, cup_traj], dim=-1).float()
        pred_z = adaptation_module(am_inputs)

        # Real push
        obs["priv_info"] = pred_z.detach().numpy()
        # obs["priv_info"] = np.random.randn(9)
        action, _ = base_policy.predict(obs)
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)

    print(np.mean(all_rewards))
from baseline2_stage_2 import AdaptationModule, get_traj
from baseline2_stage_1 import make_env
import argparse
import torch
from copy import deepcopy
from stable_baselines3 import PPO
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--am_model", type=str, required=True)
    parser.add_argument("--base_policy_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ood", action="store_true")
    args = parser.parse_args()

    adaptation_module = AdaptationModule()
    adaptation_module.load(args.am_model)
    base_policy = PPO.load(args.base_policy_model)
    
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
            e_dr_real["dynamic@floor1_table_collision@friction_sliding"] = 0.2
            e_dr_real["dynamic@floor2_table_collision@friction_sliding"] = 0.2
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
        action, _ = base_policy.predict(obs)
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)

    print(np.mean(all_rewards))
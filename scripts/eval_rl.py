from scripts.utilities import set_seed, make_env, parse_argument, generate_sample_train
from tqdm import tqdm
import torch
import numpy as np
from stable_baselines3 import PPO, SAC
def main(args):
    set_seed(args.seed)
    if args.policy_name == "ppo":
        Policy = PPO
    elif args.policy_name == "sac":
        Policy = SAC
    else:
        raise ValueError(f"Unsupported policy {args.policy_name}")
    model = Policy.load(f'output/policy_model/{args.dym_dir}/model_checkpoint_27000_steps')
    env = make_env(args.env_name, args.dr, args.sample_train)

    state, _, _ , env_params = generate_sample_train(args.data_dir, N=1000)
    rewards = []
    for idx in tqdm(range(1000)):
        obs = env.env.reset(env_params[idx], state[idx])
        action, _states = model.predict(obs, deterministic=True)
        nxt_obs, reward, done, info = env.step(action)
        rewards.append(reward)
    rewards = np.array(rewards)
    print(rewards.mean(), rewards.std())


if __name__ == "__main__":
    args = parse_argument()
    main(args)   
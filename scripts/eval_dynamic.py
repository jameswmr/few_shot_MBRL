from scripts.train_dynamic import DynamicModel
from scripts.utilities import set_seed, make_env
from tqdm import tqdm
from gym_env.utilities import ENV_CONTEXTS
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
def main(args):
    print(args.dr)
    dynamic = DynamicModel()
    dynamic.model.load_state_dict(torch.load(f'output/{args.dir}'))
    set_seed(args.seed)

    env = make_env(args.env_name, args.dr)
    state_loss = 0
    reward_loss = 0
    state_n_loss = [0 for _ in range(40)]
    times = list(range(40))
    for _ in tqdm(range(args.n_episodes)):
        obs = env.reset()
        # print(obs)
        action = env.action_space.sample()
        with torch.no_grad():
            final_state = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu()
        nxt_obs, reward, done, info = env.step(action)
        real_final_state = info['cup_pos'][-1][:2]
        print(final_state)
        print(real_final_state)
        state_loss += np.linalg.norm(real_final_state - np.array(final_state))
        reward_loss += abs(reward + np.linalg.norm(np.array(final_state) - np.array(env.target_pos[:2])))
        trajectories = env.get_state_traj(info)
        with torch.no_grad():
            for t in range(40):
                state_n = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([t], device=args.device).unsqueeze(0)).cpu()
                state_n_loss[t] += (np.linalg.norm(trajectories[t] - np.array(state_n)))
    for i in range(40):
        state_n_loss[i] /= args.n_episodes
    plt.plot(times, state_n_loss)
    plt.xlabel('Times')
    plt.ylabel('State_difference')
    plt.title('State_difference across time')
    plt.legend()
    plt.savefig('state_difference_final_state_only_38.png')
    plt.close()     
    state_loss /= args.n_episodes
    reward_loss /= args.n_episodes
    print(f'{state_loss=}')
    print(f'{reward_loss=}')
    print(f'trajectory_loss = {sum(state_n_loss)}')
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
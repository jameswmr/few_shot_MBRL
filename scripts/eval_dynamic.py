from scripts.train_dynamic import DynamicModel, RNNModel
from scripts.utilities import set_seed, make_env, parse_argument, generate_sample_train
from tqdm import tqdm
from gym_env.utilities import ENV_CONTEXTS
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt


def main(args):
    set_seed(args.seed)
    if args.method == 'rnn':
        dynamic = RNNModel(device=args.device)
        dynamic.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    else:
        dynamic = DynamicModel(device=args.device)
        dynamic.model.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    
    state, actionss, obss, env_params = generate_sample_train(args.data_dir)
    env = make_env(args.env_name, args.dr, args.sample_train)
    state_loss = 0
    reward_loss = 0
    state_n_loss = [0 for _ in range(40)]
    times = list(range(40))
    for i in tqdm(range(args.n_episodes)):
        if args.sample_train:
            
            obs = env.env.reset(env_params[i], state[i])
            # obs[2:] = [np.random.uniform(-1, 1) for _ in range(6)]
            action = actionss[i]
            # print(obs)
            # obs = env.reset()
            # print(action)
            # action = env.action_space.sample()
        else:
            action = env.action_space.sample()
            obs = env.reset()
        if args.method == "rnn":
            obs = np.repeat(obs[None, :], 40, axis=0)
            action = np.repeat(action[None, :], 40, axis=0)
        # print(obs)
        with torch.no_grad():
            dynamic.eval()
            final_state = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu().squeeze(0)
            # print(final_state.shape)
        if args.method == "mlp":
            nxt_obs, reward, done, info = env.step(action)
            real_final_state = info['cup_pos'][-1][:2]
            # print(final_state)
            # print(real_final_state)
            state_loss += np.mean((real_final_state - np.array(final_state))**2)
            reward_loss += abs(reward + np.linalg.norm(np.array(final_state) - np.array(env.target_pos[:2])))
            trajectories = env.get_state_traj(info)
            # print(trajectories[0])
            traj = []
            with torch.no_grad():
                for t in range(40):
                    state_n = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([t], device=args.device).unsqueeze(0)).cpu()
                    traj.append(state_n.flatten())
                    state_n_loss[t] += np.linalg.norm((trajectories[t] - np.array(state_n)))
            traj = torch.stack(traj)
        else:
            nxt_obs, reward, done, info = env.step(action[0])
            real_final_state = info['cup_pos'][-1][:2]
            state_loss += np.mean((real_final_state - np.array(final_state[-1]))**2)
            reward_loss += abs(reward + np.linalg.norm(np.array(final_state[-1]) - np.array(env.target_pos[:2])))
            trajectories = env.get_state_traj(info)
            for t in range(40):
                state_n_loss[t] += np.linalg.norm((trajectories[t] - np.array(final_state[t])))
            traj = final_state
        # print(traj[0])
        plt.plot(traj[:, 0], traj[:, 1], label='pred')
        plt.plot(trajectories[:, 0], trajectories[:, 1], label='real')
        plt.title(f'Trajectory')
        plt.legend()
        if args.method == "mlp":
            plt.savefig(f"images/sample_train/{np.linalg.norm((real_final_state - np.array(final_state)))}.png")
        else:
            plt.savefig(f"images/sample_train/{np.linalg.norm((real_final_state - np.array(final_state[-1])))}.png")
        plt.close()
    for i in range(40):
        state_n_loss[i] /= args.n_episodes
    np.save(f'experiment/{args.dym_dir}-large-friction_state_loss.npy',np.array(state_n_loss))
if __name__ == "__main__":
    args = parse_argument()
    main(args)   
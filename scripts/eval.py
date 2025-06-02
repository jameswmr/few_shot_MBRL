from scripts.train_dynamic import DynamicModel, RNNModel
from scripts.train_encoder import  Encoder
from scripts.utilities import set_seed, make_env, parse_argument, generate_sample_train
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
def main(args):
    if args.method == 'mlp':
        dynamic = DynamicModel(device=args.device)
        dynamic.model.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    else:
        dynamic = RNNModel(device=args.device)
        dynamic.load_state_dict(torch.load(f'output/{args.dym_dir}'))
    encoder = Encoder(device=args.device)
    encoder.load_state_dict(torch.load(f'output/{args.enc_dir}'))
    set_seed(args.seed)
    state, _, _ , env_params = generate_sample_train(args.data_dir, N=1000)
    env = make_env(args.env_name, args.dr, args.sample_train)
    
    guess_env_params = [np.random.uniform(-1, 1) for _ in range(6)]
    
    diff_params = [0 for _ in range(10)]
    rewards = []
    best_rewards = []
    for idx in tqdm(range(100)):
        for idxx in range(10):          
            obs = env.env.reset(env_params[idx], state[idx])
            real_env_params = obs[2:].copy()
            # guess_env_params = [np.random.uniform(-1, 1) for _ in range(6)]
            obs[2:] = guess_env_params
            best_action = None
            best_reward = float('-inf')
            if args.method == 'rnn':
                obs = np.repeat(obs[None, :], 40, axis=0)
            for _ in range(1000): #Parallel?
                action = env.action_space.sample()
                if args.method == 'rnn':
                    action = np.repeat(action[None, :], 40, axis=0)
                    with torch.no_grad():
                        dynamic.eval()
                        trajectories = dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([39], device=args.device).unsqueeze(0)).cpu().squeeze(0)
                    reward = -np.linalg.norm(np.array(trajectories[-1]) - np.array(env.target_pos[:2]))
                    if reward > best_reward:
                        best_action = action[0]
                        best_reward = reward
                else:
                    trajectories = []
                    with torch.no_grad():
                        for t in range(40):
                            trajectories.append(dynamic.forward(torch.tensor(obs, device=args.device).unsqueeze(0), torch.tensor(action, device=args.device).unsqueeze(0), torch.tensor([t], device=args.device).unsqueeze(0)).cpu().squeeze(0))
                    reward = -np.linalg.norm(np.array(trajectories[-1]) - np.array(env.target_pos[:2]))
                    if reward > best_reward:
                        best_action = action
                        best_reward = reward
            nxt_obs, reward, done, info = env.step(best_action)
            real_trajectories = env.get_state_traj(info)
            if args.method != 'rnn':
                trajectories = torch.stack(trajectories)
            else:
                obs = obs[0]
            with torch.no_grad():
                guess_env_params = encoder.forward(torch.tensor(trajectories, device=args.device, dtype=torch.float32).unsqueeze(0), torch.tensor(real_trajectories.reshape(80), device=args.device, dtype=torch.float32).unsqueeze(0), torch.tensor(obs, device=args.device, dtype=torch.float32).unsqueeze(0), torch.tensor(best_action, device=args.device, dtype=torch.float32).unsqueeze(0)).cpu()
            diff_params[idxx] += (np.mean((np.array(real_env_params) - np.array(guess_env_params)) ** 2))
        rewards.append(reward)
        best_rewards.append(best_reward)
    rewards = np.array(rewards)
    best_rewards = np.array(best_rewards)
    diff_params = np.array(diff_params)
    plt.plot(diff_params / 100)
    plt.xlabel('Adapatation Time')
    plt.ylabel('MSE loss')
    plt.title("Adaptation across time")
    plt.savefig(f'images/adaptation_across_time.png')
    plt.close() 
    print(rewards.mean(), rewards.std())
    print(best_rewards.mean(), best_rewards.std()) 
    # print(best_rewards - rewards)      



if __name__ == "__main__":
    args = parse_argument()
    main(args)   

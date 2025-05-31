from baseline2_stage_1 import PushExtractor, make_env
import torch
import torch.nn as nn
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import threading
import concurrent.futures

ACTION_DIM = 4
OBS_DIM = 2

class AdaptationModule(nn.Module):
    """
    Takes in the observation and the action and outputs the predicted z_t.
    """

    def __init__(self, priv_dim: int, window_size=10):
        super().__init__()

        input_dim = OBS_DIM + (ACTION_DIM + 1) * window_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, priv_dim)
        )
    
    def forward(self, p_t):
        return torch.tanh(self.encoder(p_t))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def generate_one_sample(task_name, seed, window_size):
    headless = True
    state_dim = 2
    priv_dim = 9

    env = make_env(task_name, seed, headless, state_dim, priv_dim)()
    obs = env.reset()
    am_inputs = [obs["obs"]]
    for j in range(window_size):
        sim_env = deepcopy(env)
        rand_action = sim_env.action_space.sample()
        _, reward, _, _ = sim_env.step(rand_action)
        am_inputs.extend([rand_action, np.array([reward])])
    am_inputs = np.concatenate(am_inputs, axis=0)
    am_inputs = torch.from_numpy(am_inputs).float()
    return (am_inputs, obs["priv_info"])

if __name__ == "__main__":
    # Load the pretrained policy model and freeze it
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy_path", type=str, required=True)
    parser.add_argument("--task_name", default="push_one_task")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/baseline2_stage_2")
    parser.add_argument("--training_set_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # Load the base policy model and freeze it
    base_policy = PPO.load(args.base_policy_path)
    push_extractor = base_policy.policy.features_extractor
    
    set_random_seed(args.seed)

    # Build environment
    headless = True
    state_dim = 2
    priv_dim = 9

    # Initialize the adaptation module
    training_set = []

    # Run the game for 1 episodes
    all_rewards = []
    pbar = tqdm(range(args.training_set_size), desc="Collecting training set")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(generate_one_sample, args.task_name, args.seed, args.window_size) for _ in range(args.training_set_size)]
        for future in concurrent.futures.as_completed(futures):
            training_set.append(future.result())
            pbar.update(1)

    # Train the adaptation module
    adaptation_module = AdaptationModule(priv_dim, window_size=args.window_size)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(args.num_epochs), desc="Training adaptation module")
    for _ in pbar:
        for am_inputs, gt_z in train_loader:
            pred_z = adaptation_module(am_inputs)
            loss = criterion(pred_z, gt_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
    
    # Save the adaptation module
    adaptation_module.save(os.path.join(args.output_dir, "adaptation_module.pth"))

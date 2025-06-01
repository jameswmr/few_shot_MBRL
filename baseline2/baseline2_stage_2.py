from few_shot_MBRL.baseline2.baseline2_stage_1 import PushExtractor, make_env
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
import pickle

ACTION_DIM = 4
OBS_DIM = 2
TRAJ_DIM = 40 * 2
PRIV_DIM = 9
AM_INPUT_DIM = OBS_DIM + ACTION_DIM + TRAJ_DIM

class AdaptationModule(nn.Module):
    """
    Takes in the observation and the action and outputs the predicted z_t.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(AM_INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, PRIV_DIM)
        )
    
    def forward(self, p_t):
        return torch.tanh(self.encoder(p_t))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def get_traj(info):
    state_traj = []
    for index in range(len(info["cup_pos"])):
        trajectory = info["cup_pos"][index][:2]
        trajectory = np.array(trajectory).flatten()
        state_traj.append(trajectory)
    return np.array(state_traj)


def generate_one_sample(task_name, seed):
    headless = True
    state_dim = 2
    priv_dim = 9

    env = make_env(task_name, seed, headless, state_dim, priv_dim)()
    obs = env.reset()
    cup_pos = obs["obs"]
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cup_traj = get_traj(info)
    return (cup_pos, action, cup_traj, obs["priv_info"])


if __name__ == "__main__":
    # Load the pretrained policy model and freeze it
    parser = argparse.ArgumentParser()
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

    os.makedirs(args.output_dir, exist_ok=True)

    set_random_seed(args.seed)

    if os.path.exists(os.path.join(args.output_dir, f"training_set_{args.training_set_size}.pkl")):
        # Load training set
        with open(os.path.join(args.output_dir, f"training_set_{args.training_set_size}.pkl"), "rb") as f:
            training_set = pickle.load(f)
    else:
        # Collect training set
        training_set = []
        pbar = tqdm(range(args.training_set_size), desc="Collecting training set")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(generate_one_sample, args.task_name, args.seed) for _ in range(args.training_set_size)]
            for future in concurrent.futures.as_completed(futures):
                training_set.append(future.result())
                pbar.update(1)
        
        with open(os.path.join(args.output_dir, f"training_set_{args.training_set_size}.pkl"), "wb") as f:
            pickle.dump(training_set, f)

    # Train the adaptation module
    adaptation_module = AdaptationModule()
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(args.num_epochs), desc="Training adaptation module")
    for _ in pbar:
        for cup_pos, action, cup_traj, priv_info in train_loader:

            # Reshape am_inputs to (batch_size, AM_INPUT_DIM)
            cup_traj = cup_traj.reshape(-1, TRAJ_DIM)
            am_inputs = torch.cat([cup_pos, action, cup_traj], dim=-1).float()

            # Forward pass
            pred_z = adaptation_module(am_inputs)
            loss = criterion(pred_z, priv_info)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
    
    # Save the adaptation module
    adaptation_module.save(os.path.join(args.output_dir, "adaptation_module.pth"))

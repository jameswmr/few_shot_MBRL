import gym
from gym import spaces
import numpy as np
import torch
from scripts.train_dynamic import RNNModel
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )
        self.dynamic = RNNModel()
        self.dynamic.load_state_dict(torch.load('output/best_rnn_32.pt'))

    def reset(self):
        self.obs = np.array([np.random.uniform(-1, 1) for _ in range(8)], dtype=np.float32)
        return self.obs
    def step(self, action):
        obs = np.repeat(self.obs[None, :], 40, axis=0).astype(np.float32) 
        action = np.repeat(action[None, :], 40, axis=0).astype(np.float32)
        with torch.no_grad():
            self.dynamic.eval()
            trajectories = self.dynamic.forward(torch.tensor(obs, device="cuda:0", dtype=torch.float32).unsqueeze(0), torch.tensor(action, device="cuda:0", dtype=torch.float32).unsqueeze(0), torch.tensor([39], device="cuda:0", dtype=torch.float32).unsqueeze(0)).cpu().squeeze(0)
        reward = -np.linalg.norm(np.array(trajectories[-1]) - np.array([0.29, 0]))
        info = {
            "episode": {
                "r": reward,
                "l": 40,
            }
        }
        return self.obs, reward, True, info
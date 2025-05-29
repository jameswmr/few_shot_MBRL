#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import os
from typing import Callable, Dict, Any

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback  # type: ignore
except ImportError:  # wandb is optional
    wandb = None  # type: ignore
    WandbCallback = None  # type: ignore


class SB3Stage1Wrapper(gym.Wrapper):
    """Wrap task so that obs → Dict and arrays are np.float32."""

    def __init__(self, env: gym.Env, state_dim: int, priv_dim: int):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
                "priv_info": spaces.Box(low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._convert(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._convert(obs), rew, done, info

    @staticmethod
    def _convert(obs_dict: Dict[str, Any]):
        conv = (
            lambda x: x.detach().cpu().numpy().astype(np.float32)
            if isinstance(x, torch.Tensor)
            else x.astype(np.float32)
        )
        return {k: conv(v) for k, v in obs_dict.items()}


class PushExtractor(BaseFeaturesExtractor):
    """Encode priv‑info → zₜ, concatenate with obs, then MLP."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        state_dim = observation_space["obs"].shape[0]
        priv_dim = observation_space["priv_info"].shape[0]
        latent_dim = 8  # dimension of z_t

        # μ encoder
        self.encoder = nn.Sequential(
            nn.Linear(priv_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )

        self._features_dim = 128

    def forward(self, obs):
        o_t = obs["obs"]
        p_t = obs["priv_info"]
        z_t = torch.tanh(self.encoder(p_t))
        x = torch.cat([o_t, z_t], dim=-1)
        return self.backbone(x)


def make_env(task_name: str, seed: int, headless: bool, state_dim: int, priv_dim: int) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped environment."""

    def _init():
        try:
            from few_shot_MBRL.baseline2.tasks import isaacgym_task_map
            TaskCls = isaacgym_task_map[task_name]
        except (ImportError, KeyError):
            from few_shot_MBRL.baseline2.envs.push_one_task import PushOneTask as TaskCls
        env = TaskCls(sim_device="cpu", graphics_device_id=0, headless=headless)
        env.seed(seed)
        env = SB3Stage1Wrapper(env, state_dim, priv_dim)
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task_name", default="push_one_task")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save_path", default="./output/stage1_sb3/base_policy")
    # observation dims (set according to the env)
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--priv_dim", type=int, default=9)
    # wandb options
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    set_random_seed(args.seed)

    env_thunk = lambda _seed: make_env(
        args.task_name, _seed, args.headless, args.state_dim, args.priv_dim
    )
    if args.n_envs == 1:
        env = DummyVecEnv([env_thunk(args.seed)])
    else:
        env = SubprocVecEnv([env_thunk(args.seed + i) for i in range(args.n_envs)])

    policy_kwargs = dict(
        features_extractor_class=PushExtractor,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        device=args.device,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16 * args.n_envs,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        tensorboard_log=os.path.join(os.path.dirname(args.save_path), "tb"),
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=1,
    )

    callback = None
    run = None
    if args.wandb:
        if wandb is None or WandbCallback is None:
            raise ImportError("wandb not installed: pip install wandb")
        run = wandb.init(
            project=f"train_task_policy_push_one",
            name=f'stage_1_ppo-{time.strftime("%Y-%m-%d-%H-%M")}',
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )
        callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=os.path.dirname(args.save_path),
            verbose=2,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"\nPolicy saved to {args.save_path}.zip")

    if run is not None:
        run.finish()

    env.close()


if __name__ == "__main__":
    main()

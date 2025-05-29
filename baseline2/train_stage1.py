#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import gym
import numpy as np
import torch

class AttrDict(dict):
    """dict that also supports attribute access (x.y)."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def make_config(
    device: str,
    out_dir: str,
    num_steps: int,
    *,
    horizon_length: int = 16,
    num_actors: int = 1,
) -> AttrDict:
    """Return an AttrDict configuration understood by PPO."""

    batch_size = horizon_length * num_actors
    # pick a minibatch_size that divides batch_size
    minibatch_size = batch_size

    cfg = AttrDict(
        {
            "rl_device": device,
            "train": SimpleNamespace(
                network=SimpleNamespace(
                    mlp=SimpleNamespace(units=[512, 256, 128]),
                    priv_mlp=SimpleNamespace(units=[256, 128, 8]),
                ),
                ppo={
                    "output_name": os.path.basename(out_dir),
                    "normalize_input": True,
                    "normalize_value": True,
                    "value_bootstrap": True,
                    "num_actors": num_actors,
                    "normalize_advantage": True,
                    "gamma": 0.99,
                    "tau": 0.95,
                    "learning_rate": 3e-4,
                    "kl_threshold": 0.02,
                    "horizon_length": horizon_length,
                    "minibatch_size": minibatch_size,
                    "mini_epochs": 5,
                    "clip_value": True,
                    "critic_coef": 4.0,
                    "entropy_coef": 0.0,
                    "e_clip": 0.2,
                    "bounds_loss_coef": 1e-4,
                    "truncate_grads": True,
                    "grad_norm": 1.0,
                    "save_best_after": 0,
                    "save_frequency": 500,
                    "max_agent_steps": num_steps,
                    # -------- Stage‑1 specific flags --------
                    "priv_info": True,
                    "priv_info_dim": 9,
                    "priv_info_embed_dim": 8,
                    "proprio_adapt": False,
                },
            ),
        }
    )
    return cfg


class DeviceWrapper(gym.Wrapper):
    """Move all tensors produced by the env to the desired device.
    Also converts torch actions ➔ numpy before passing into the raw env,
    and back‑to‑torch for the reward / done flags.
    """

    def __init__(self, env: gym.Env, device: str):
        super().__init__(env)
        self.device = device

    def _to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def reset(self):
        obs = self.env.reset()
        return self._to_device(obs)

    def step(self, action):
        # make sure the action is numpy for the underlying env
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        obs, reward, done, info = self.env.step(action)
        # convert outputs to torch on the right device
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        obs = self._to_device(obs)
        return obs, reward, done, info



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="cuda:0 | cpu")
    parser.add_argument("--out_dir", default="./output/stage1")
    parser.add_argument("--num_steps", type=int, default=1_000_000)
    parser.add_argument("--task_name", default="push_one_task")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = make_config(args.device, args.out_dir, args.num_steps)

    try:
        from few_shot_MBRL.baseline2.tasks import isaacgym_task_map
        TaskCls = isaacgym_task_map[args.task_name]
    except (ImportError, KeyError):
        # fall‑back to PushOneTask if task map is missing
        from few_shot_MBRL.baseline2.env.push_one_task import PushOneTask as TaskCls

    raw_env = TaskCls(sim_device=args.device, graphics_device_id=0, headless=args.headless)
    env = DeviceWrapper(raw_env, args.device)
    # def make_env(seed):
    #     def _thunk():
    #         env = TaskCls(sim_device=args.device, graphics_device_id=0, headless=args.headless)
    #         env.seed(seed)
    #         return DeviceWrapper(env, args.device)
    #     return _thunk

    # if args.num_actors > 1:
    # env = gym.vector.SyncVectorEnv([make_env(s) for s in range(8)])
    # else:
    #     env = DeviceWrapper(raw_env, args.device)

    from few_shot_MBRL.baseline2.ppo.ppo import PPO

    agent = PPO(env, args.out_dir, cfg)
    agent.train()


if __name__ == "__main__":
    main()

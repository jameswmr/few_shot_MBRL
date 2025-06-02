from __future__ import annotations

import os
import time
from typing import Any, Dict

import gym
import torch
from tensorboardX import SummaryWriter

from few_shot_MBRL.baseline2.ppo.experience import ExperienceBuffer
from few_shot_MBRL.baseline2.models.models import ActorCritic
from few_shot_MBRL.baseline2.models.running_mean_std import RunningMeanStd
from few_shot_MBRL.baseline2.utils.misc import AverageScalarMeter

__all__ = ["PPO"]


# ------------------------------------------------------------------------- #
# Helper functions / classes
# ------------------------------------------------------------------------- #
def _ensure_tensor(x, dtype, device) -> torch.Tensor:
    """Cast scalar / list / numpy / Tensor to shape = (B,) Tensor on device."""
    if isinstance(x, torch.Tensor):
        x = x.to(device=device, dtype=dtype)
        if x.dim() == 0:            # <─ 新增：處理 0-D tensor
            x = x.unsqueeze(0)      # 變成 shape (1,)
        return x
    return torch.tensor([x], dtype=dtype, device=device)


def _policy_kl(mu0, sigma0, mu1, sigma1):
    kl = (
        torch.log(sigma1 / sigma0 + 1e-5)
        + (sigma0**2 + (mu1 - mu0) ** 2) / (2.0 * (sigma1**2 + 1e-5))
        - 0.5
    )
    return kl.sum(dim=-1).mean()


class AdaptiveScheduler:
    def __init__(self, kl_threshold: float = 0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, lr: float, kl: float):
        if kl > 2.0 * self.kl_threshold:
            lr = max(lr / 1.5, self.min_lr)
        elif kl < 0.5 * self.kl_threshold:
            lr = min(lr * 1.5, self.max_lr)
        return lr


# ------------------------------------------------------------------------- #
# PPO main
# ------------------------------------------------------------------------- #
class PPO:
    """Stage-1 PPO trainer adapted for Push-One task."""

    # --------------------------- init ----------------------------------- #
    def __init__(self, env: gym.Env, output_dir: str, full_cfg: Dict[str, Any]):
        # devices & cfg
        self.device: str = full_cfg["rl_device"]
        net_cfg = full_cfg.train.network
        ppo_cfg = full_cfg.train.ppo
        self.ppo_cfg = ppo_cfg

        # env
        self.env = env
        self.num_actors = ppo_cfg["num_actors"]
        act_space = env.action_space
        self.act_dim = act_space.shape[0]
        self.act_low = torch.as_tensor(act_space.low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_space.high, dtype=torch.float32, device=self.device)
        self.obs_shape = env.observation_space.shape  # (obs_dim,)
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_shape = obs_space["obs"].shape
        else:
            self.obs_shape = obs_space.shape 

        # model
        model_cfg = dict(
            actor_units=net_cfg.mlp.units,
            priv_mlp_units=net_cfg.priv_mlp.units,
            actions_num=self.act_dim,
            input_shape=self.obs_shape,
            priv_info=ppo_cfg["priv_info"],
            proprio_adapt=ppo_cfg["proprio_adapt"],
            priv_info_dim=ppo_cfg["priv_info_dim"],
        )
        self.model = ActorCritic(model_cfg).to(self.device)
        self.obs_ms = RunningMeanStd(self.obs_shape).to(self.device)
        self.val_ms = RunningMeanStd((1,)).to(self.device)

        # dirs / writer
        self.nn_dir = os.path.join(output_dir, "stage1_nn")
        self.tb_dir = os.path.join(output_dir, "stage1_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # optim
        self.lr = float(ppo_cfg["learning_rate"])
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=ppo_cfg.get("weight_decay", 0.0)
        )

        # hyper-params
        self.e_clip = ppo_cfg["e_clip"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.critic_coef = ppo_cfg["critic_coef"]
        self.bounds_coef = ppo_cfg["bounds_loss_coef"]
        self.gamma = ppo_cfg["gamma"]
        self.tau = ppo_cfg["tau"]
        self.truncate_grads = ppo_cfg["truncate_grads"]
        self.grad_norm = ppo_cfg["grad_norm"]
        self.value_bootstrap = ppo_cfg["value_bootstrap"]
        self.norm_adv = ppo_cfg["normalize_advantage"]
        self.norm_obs = ppo_cfg["normalize_input"]
        self.norm_val = ppo_cfg["normalize_value"]

        # rollout sizes
        self.horizon = ppo_cfg["horizon_length"]
        self.batch = self.horizon * self.num_actors
        self.minibatch = ppo_cfg["minibatch_size"]
        self.mini_epochs = ppo_cfg["mini_epochs"]
        assert self.batch % self.minibatch == 0 or full_cfg.test, "minibatch must divide batch"

        # schedule / snapshots
        self.scheduler = AdaptiveScheduler(ppo_cfg["kl_threshold"])
        self.save_freq = ppo_cfg["save_frequency"]
        self.save_best_after = ppo_cfg["save_best_after"]

        # meters
        self.ep_ret = AverageScalarMeter(100)
        self.ep_len = AverageScalarMeter(100)

        # buffer
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon,
            self.batch,
            self.minibatch,
            self.obs_shape[0],
            self.act_dim,
            ppo_cfg["priv_info_dim"],
            self.device,
        )

        # rollout states
        self.obs = None  # dict
        self.current_ret = torch.zeros((self.num_actors, 1), device=self.device)
        self.current_len = torch.zeros(self.num_actors, device=self.device)
        self.dones = torch.ones(self.num_actors, dtype=torch.uint8, device=self.device)

        self.agent_steps = 0
        self.max_agent_steps = ppo_cfg["max_agent_steps"]
        self.best_reward = -1e9

        self.collect_time = 0.0
        self.train_time = 0.0
        self.epoch = 0
        self.extra_info: Dict[str, Any] = {}

    # ---------------------- public API ---------------------------------- #
    def train(self):
        start = last = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            a_l, c_l, b_l, ent, kl = self._train_epoch()
            self.storage.data_dict = None  # free GPU mem

            fps_total = self.agent_steps / (time.time() - start + 1e-8)
            fps_last = self.batch / (time.time() - last + 1e-8)
            last = time.time()
            print(
                f"Epoch {self.epoch:03d} | Steps {self.agent_steps/1e6:5.5f}M | "
                f"FPS {fps_total:7.1f} / {fps_last:7.1f} | "
                f"Collect {self.collect_time/60:5.5f}m | Train {self.train_time/60:5.5f}m | "
                f"Best {self.best_reward:7.7f}"
            )
            self._write_stats(a_l, c_l, b_l, ent, kl)

            # snapshots
            mean_rew = self.ep_ret.get_mean()
            print (mean_rew)
            if self.save_freq > 0 and self.epoch % self.save_freq == 0:
                name = f"ep{self.epoch:03d}_step{int(self.agent_steps/1e6):04}M_rew{mean_rew:.2f}"
                self._save(os.path.join(self.nn_dir, name))
                self._save(os.path.join(self.nn_dir, "last"))
            if mean_rew > self.best_reward and self.epoch >= self.save_best_after:
                self.best_reward = mean_rew
                self._save(os.path.join(self.nn_dir, "best"))

        print("Training finished – reached max_agent_steps.")

    # ----------------------- epoch -------------------------------------- #
    def _train_epoch(self):
        # -------- rollout --------
        t0 = time.time()
        self._collect_rollout()
        self.collect_time += time.time() - t0

        # -------- update ---------
        t0 = time.time()
        self.model.train(); self.obs_ms.train(); self.val_ms.train()
        a_losses, c_losses, b_losses, entropies, kls = [], [], [], [], []

        for _ in range(self.mini_epochs):
            epoch_kls = []
            for i in range(len(self.storage)):
                (v_pred, old_logp, adv, old_mu, old_sigma,
                 returns, actions, obs, priv) = self.storage[i]

                obs = self.obs_ms(obs)
                res = self.model({"prev_actions": actions, "obs": obs, "priv_info": priv})
                logp = res["prev_neglogp"]
                values = res["values"]
                entropy = res["entropy"]
                mu, sigma = res["mus"], res["sigmas"]

                # actor loss
                ratio = torch.exp(old_logp - logp)
                surr1 = adv * ratio
                surr2 = adv * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)

                # critic loss
                v_clip = v_pred + (values - v_pred).clamp(-self.e_clip, self.e_clip)
                v_loss1 = (values - returns) ** 2
                v_loss2 = (v_clip - returns) ** 2
                c_loss = torch.max(v_loss1, v_loss2)

                # bounds loss
                if self.bounds_coef > 0:
                    soft_b = 1.1
                    b_high = torch.clamp_min(mu - soft_b, 0.0) ** 2
                    b_low = torch.clamp_min(-mu + soft_b, 0.0) ** 2
                    b_loss = (b_high + b_low).sum(dim=-1)
                else:
                    b_loss = torch.zeros_like(a_loss)

                a_loss, c_loss, entropy, b_loss = [x.mean() for x in (a_loss, c_loss, entropy, b_loss)]
                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_coef
                )

                self.opt.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = _policy_kl(mu, sigma, old_mu, old_sigma)

                epoch_kls.append(kl)
                a_losses.append(a_loss); c_losses.append(c_loss); b_losses.append(b_loss); entropies.append(entropy)
                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            mean_kl = torch.stack(epoch_kls).mean()
            self.lr = self.scheduler.update(self.lr, mean_kl.item())
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            kls.append(mean_kl)

        self.train_time += time.time() - t0
        print (self.train_time)
        return a_losses, c_losses, b_losses, entropies, kls

    # ----------------------- rollout ------------------------------------ #
    def _collect_rollout(self):
        self.model.eval(); self.obs_ms.eval(); self.val_ms.eval()
        for t in range(self.horizon):
            if torch.all(self.dones).item():                        
                self.obs = self.env.reset()
                self.dones = torch.zeros_like(self.dones)      
                self.current_ret.zero_()
                self.current_len.zero_()
                
            res = self._model_act(self.obs)

            # store obs / priv
            self.storage.update_data("obses", t, self.obs["obs"])
            self.storage.update_data("priv_info", t, self.obs["priv_info"])
            for k in ("actions", "neglogpacs", "values", "mus", "sigmas"):
                self.storage.update_data(k, t, res[k])

            # env step
            act_env = torch.clamp(res["actions"], -1.0, 1.0)
            self.obs, r, self.dones, info = self.env.step(act_env)

            # make sure r, dones are tensors (B,)
            r = _ensure_tensor(r, torch.float32, self.device)
            dones = _ensure_tensor(self.dones, torch.uint8, self.device)
            r = r.unsqueeze(1)  # (B,1)

            # store rewards / dones
            self.storage.update_data("dones", t, dones)
            shaped_r = 0.01 * r.clone()
            if self.value_bootstrap and isinstance(info, dict) and "time_outs" in info:
                shaped_r += self.gamma * res["values"] * info["time_outs"].unsqueeze(1).float()
            self.storage.update_data("rewards", t, shaped_r)

            # episode meters
            self.current_ret += r
            self.current_len += 1
            done_idx = dones.nonzero(as_tuple=False)
            if done_idx.numel():
                self.ep_ret.update(self.current_ret[done_idx].view(-1))
                self.ep_len.update(self.current_len[done_idx])

            not_dones = 1.0 - dones.float()
            self.current_ret *= not_dones.unsqueeze(1)
            self.current_len *= not_dones

            # optional extra scalars
            if isinstance(info, dict):
                self.extra_info = {
                    k: v for k, v in info.items() if isinstance(v, (float, int)) or (isinstance(v, torch.Tensor) and v.dim() == 0)
                }

        # bootstrap value
        last_val = self._model_act(self.obs)["values"]
        self.agent_steps += self.batch
        self.storage.computer_return(last_val, self.gamma, self.tau)
        self.storage.prepare_training()

        if self.norm_val:
            self.val_ms.train()
            self.storage.data_dict["values"] = self.val_ms(self.storage.data_dict["values"])
            self.storage.data_dict["returns"] = self.val_ms(self.storage.data_dict["returns"])
            self.val_ms.eval()

    # ----------------------- model act ---------------------------------- #
    def _model_act(self, obs_dict):
        obs_norm = self.obs_ms(obs_dict["obs"])
        res = self.model.act({
            "obs":        obs_norm,
            "priv_info":  obs_dict["priv_info"]
        })
        res["values"] = self.val_ms(res["values"], True)
        return res

    # ----------------------- utils -------------------------------------- #
    def _write_stats(self, a_l, c_l, b_l, ent, kl):
        step = self.agent_steps
        self.writer.add_scalar("performance/RL_fps", step / max(self.train_time, 1e-6), step)
        self.writer.add_scalar("performance/Env_fps", step / max(self.collect_time, 1e-6), step)

        self.writer.add_scalar("loss/actor", torch.stack(a_l).mean().item(), step)
        self.writer.add_scalar("loss/critic", torch.stack(c_l).mean().item(), step)
        self.writer.add_scalar("loss/bounds", torch.stack(b_l).mean().item(), step)
        self.writer.add_scalar("loss/entropy", torch.stack(ent).mean().item(), step)
        self.writer.add_scalar("info/kl", torch.stack(kl).mean().item(), step)
        self.writer.add_scalar("info/lr", self.lr, step)
        for k, v in self.extra_info.items():
            self.writer.add_scalar(f"env/{k}", v, step)

        self.writer.add_scalar("episode/rew_mean", self.ep_ret.get_mean(), step)
        self.writer.add_scalar("episode/len_mean", self.ep_len.get_mean(), step)

    # ----------------------- save --------------------------------------- #
    def _save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_ms": self.obs_ms.state_dict(),
                "val_ms": self.val_ms.state_dict(),
            },
            f"{path}.pth",
        )

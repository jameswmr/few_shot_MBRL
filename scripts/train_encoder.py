import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import wandb
from torch.utils.data import Dataset
from datetime import datetime
import os
from scripts.train_dynamic import RNNModel

import torch

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.xy_mean = 0
        self.xy_std = 1
        self.action_mean = 0
        self.action_std = 1
        self.env_mean = 0
        self.env_std = 1

    def fit(self, trajectory, real_trajectory, obs, actions, env_params):
        """
        Compute and store the mean/std of inputs.
        trajectory, real_trajectory: (N, 40, 2)
        obs: (N, 8)
        actions: (N, 4)
        env_params: (N, 6)
        """
        traj_all = torch.cat([trajectory.view(-1, 2), real_trajectory.view(-1, 2)], dim=0)
        self.xy_mean = traj_all.mean(0)
        self.xy_std = traj_all.std(0) + 1e-6

        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0) + 1e-6

        self.env_mean = env_params.mean(0)
        self.env_std = env_params.std(0) + 1e-6

    def normalize_trajectory(self, traj):  # shape: (40, 2) or (B, 40, 2)
        B = traj.size(0)
        return ((traj.view(-1, 2) - self.xy_mean) / self.xy_std).view(B, -1)

    def denormalize_trajectory(self, traj_norm):
        B = traj_norm.size(0)
        return (traj_norm.view(-1, 2) * self.xy_std + self.xy_mean).view(B, -1)

    def normalize_obs(self, obs):  # shape: (8,) or (B, 8)
        xy = (obs[..., :2] - self.xy_mean) / self.xy_std
        env = (obs[..., 2:] - self.env_mean) / self.env_std
        return torch.cat([xy, env], dim=-1)

    def denormalize_obs(self, obs_norm):
        xy = obs_norm[..., :2] * self.xy_std + self.xy_mean
        env = obs_norm[..., 2:] * self.env_std + self.env_mean
        return torch.cat([xy, env], dim=-1)

    def normalize_action(self, action):  # shape: (4,) or (B, 4)
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action_norm):
        return action_norm * self.action_std + self.action_mean

    def normalize_env_params(self, env):  # shape: (6,) or (B, 6)
        return (env - self.env_mean) / self.env_std

    def denormalize_env_params(self, env_norm):
        return env_norm * self.env_std + self.env_mean

class EncoderDataset(Dataset):
    def __init__(self, data_dict):
        size = len(data_dict['obs'])
        self.trajectory = data_dict['trajectory'].reshape(size, -1)
        self.real_trajectory = data_dict['real_trajectory'].reshape(size, -1)
        self.obs = data_dict['obs']
        self.actions = data_dict['actions']
        self.env_params = data_dict['env_params']

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            self.trajectory[idx],
            self.real_trajectory[idx],
            self.obs[idx],
            self.actions[idx],
            self.env_params[idx],
        )

    def set_normalizer(self, normalizer:Normalizer):
        normalizer.fit(self.trajectory, self.real_trajectory, self.obs, self.actions, self.env_params)
        
class Encoder(nn.Module):
    def __init__(self,
        obs_dim = 8,
        act_dim = 4,
        output_dim = 6,
        trajectory_dim = 40,
        hidden_dim = 256,
        device = "cuda",
        shot_steps = 10,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        input_dim = 128 * 10 + obs_dim + act_dim
        self.mean_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.device = device
        self.to(device)

        self.normalizer = Normalizer()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_param_dim = output_dim
        self.traj_dim = trajectory_dim
        self.hidden_dim = hidden_dim
        self.shot_steps = shot_steps

    def forward(self, pred_traj, true_traj, obs, action, normalize=True):
        B = pred_traj.size(0)

        # Correct reshape: (B, 80) → (B, 40, 2) → (B, 2, 40)
        pred = pred_traj.view(B, 40, 2).permute(0, 2, 1)  # (B, 2, 40)
        true = true_traj.view(B, 40, 2).permute(0, 2, 1)  # (B, 2, 40)
        diff = pred - true

        x = torch.cat([diff, pred, true], dim=1)
        x = self.conv(x)
        x = torch.cat([x, obs, action], dim=1)
        mean = self.mean_layer(x)
        std = torch.clamp(self.log_std_layer(x), -20, 2).exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()

        if normalize:
            x = self.normalizer.denormalize_env_params(x)

        return x

    def train(self,   
            dynamic_model: RNNModel,
            lr = 1e-3,
            data_dir = None,
            epochs = 2000,
            batch_size = 256,
            eval_interval = 20,
        ):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        dataset = EncoderDataset(torch.load(f'data/{data_dir}'))

        # dataset.set_normalizer(self.normalizer)

        run_name = f"encoder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        output_dir = f"./output/{run_name}"
        logger = wandb.init(
            project='ev_drl',
            name=run_name,
            dir=output_dir
        )
        os.mkdir(os.path.join(output_dir, "checkpoint/"))

        best_val_loss = float('inf')
        best_model_state = None

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        global_step = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
        loss_weights = ((torch.range(1, self.shot_steps) / self.shot_steps) - 1).exp().to(self.device)
        loss_weights_sum = loss_weights.sum()

        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            for traj, real_traj, obs, action, env_params in train_loader:
                # normalization
                real_traj = self.normalizer.normalize_trajectory(real_traj)
                obs = self.normalizer.normalize_obs(obs)
                action = self.normalizer.normalize_action(action)
                env_params = self.normalizer.normalize_env_params(env_params)

                real_traj = real_traj.to(self.device).float()
                obs = obs.to(self.device).float()
                action = action.to(self.device).float()
                env_params = env_params.to(self.device).float()

                losses = []
                for i in range(self.shot_steps):
                    traj = dynamic_model.forward(obs, action, time=self.traj_dim).view(real_traj.shape[0], -1)
                    traj = self.normalizer.normalize_trajectory(traj)
                    traj = traj.to(self.device).float()

                    pred_env_params = self.forward(traj, real_traj, obs, action, normalize=False)
                    loss = loss_fn(pred_env_params, env_params)
                    losses.append(loss)

                    obs[..., -6:] = pred_env_params

                loss = (torch.stack(losses) * loss_weights).sum() / loss_weights_sum
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                raw_loss = loss.cpu().detach().item()
                total_train_loss += raw_loss * obs.size(0)
                logger.log({
                    "train/loss": raw_loss,
                    "train/global_step": global_step,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)
                global_step += 1

            avg_train_loss = total_train_loss / len(train_loader.dataset)


            if epoch % eval_interval == 0:
                total_val_loss = 0
                with torch.no_grad():
                    for traj, real_traj, obs, action, env_params in val_loader:
                        # normalization
                        traj = self.normalizer.normalize_trajectory(traj)
                        real_traj = self.normalizer.normalize_trajectory(real_traj)
                        obs = self.normalizer.normalize_obs(obs)
                        action = self.normalizer.normalize_action(action)
                        env_params = self.normalizer.normalize_env_params(env_params)

                        traj = traj.to(self.device).float()
                        real_traj = real_traj.to(self.device).float()
                        obs = obs.to(self.device).float()
                        action = action.to(self.device).float()
                        env_params = env_params.to(self.device).float()

                        losses = []
                        for i in range(self.shot_steps):
                            traj = dynamic_model.forward(obs, action, time=self.traj_dim)
                            traj = self.normalizer.normalize_trajectory(traj)
                            traj = traj.to(self.device).float()

                            pred_env_params = self.forward(traj, real_traj, obs, action, normalize=False)
                            loss = loss_fn(pred_env_params, env_params)
                            losses.append(loss)

                            obs[:, -6:] = pred_env_params

                        loss = (torch.stack(losses) * loss_weights).sum() / loss_weights_sum
                        raw_loss = loss.item()
                        total_val_loss += raw_loss * obs.size(0)

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                logger.log({
                    "eval/loss": avg_val_loss,
                }, step = global_step)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.state_dict()
                    torch.save(best_model_state, os.path.join(output_dir, "checkpoint/", f"epoch_{epoch}.ckpt"))
                    print(">> Saved best encoder")
            
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f}")

        torch.save(best_model_state, os.path.join(output_dir, "checkpoint/", f"latest.ckpt"))
        print(">> Saved latest encoder")
        print("Training finished. Best validation loss:", best_val_loss)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data_encoder_full.pt')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-interval', type=int, default=20)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--shot-steps', type=int, default=10)
    parser.add_argument('--dynamic-model-ckpt', type=str, default="output/LSTM-checkpoint/best_rnn_32.pt")
    args = parser.parse_args()

    dynamic_model = RNNModel()
    dynamic_model.load_state_dict(torch.load(args.dynamic_model_ckpt, map_location='cuda'))

    encoder = Encoder(hidden_dim=args.hidden_dim, shot_steps=args.shot_steps)
    encoder.train(
        dynamic_model = dynamic_model,
        lr = args.lr,
        data_dir = args.data_dir,
        epochs = args.epochs,
        batch_size = args.batch_size,
        eval_interval = args.eval_interval,
    )

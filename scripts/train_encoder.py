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

class Encoder(nn.Module):
    def __init__(self,
        obs_dim = 8,
        act_dim = 4,
        output_dim = 6,
        trajectory_dim = 40,
        hidden_dim = 256,
        device = "cuda"
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
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.device = device
        self.conv.to(self.device)
        self.fc.to(self.device)

    def forward(self, pred_traj, true_traj, obs, action):
        B = pred_traj.size(0)

        # Preprocess trajectories: (B, 80) → (B, 2, 40)
        pred = pred_traj.view(B, 2, 40)
        true = true_traj.view(B, 2, 40)
        diff = pred - true

        x = torch.cat([diff, pred, true], dim=1)
        x = self.conv(x)
        x = torch.cat([x, obs, action], dim=1)
        x = self.fc(x)
        return x

    def train(self,   
            lr = 1e-3,
            data_dir = None,
            epochs = 2000,
            batch_size = 256,
            eval_interval = 20,
        ):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        dataset = EncoderDataset(torch.load(f'data/{data_dir}'))

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

        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            for traj, real_traj, obs, action, env_params in train_loader:
                traj = traj.to(self.device).float()
                real_traj = real_traj.to(self.device).float()
                obs = obs.to(self.device).float()
                action = action.to(self.device).float()
                env_params = env_params.to(self.device).float()

                pred_env_params = self.forward(traj, real_traj, obs, action)
                loss = loss_fn(pred_env_params, env_params)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                raw_loss = loss.cpu().detach().item()
                total_train_loss += raw_loss * obs.size(0)
                logger.log({
                    "train/loss": raw_loss,
                })

            avg_train_loss = total_train_loss / len(train_loader.dataset)


            if epoch % eval_interval == 0:
                total_val_loss = 0
                with torch.no_grad():
                    for traj, real_traj, obs, action, env_params in train_loader:
                        traj = traj.to(self.device).float()
                        real_traj = real_traj.to(self.device).float()
                        obs = obs.to(self.device).float()
                        action = action.to(self.device).float()
                        env_params = env_params.to(self.device).float()

                        pred_env_params = self.forward(traj, real_traj, obs, action)
                        loss = loss_fn(pred_env_params, env_params)
                        raw_loss = loss.item()
                        total_val_loss += raw_loss * obs.size(0)
                        logger.log({
                            "eval/loss": raw_loss,
                        })

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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

    parser.add_argument('--data_dir', type=str, default='data_encoder.pt')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-interval', type=int, default=20)
    args = parser.parse_args()

    encoder = Encoder()
    encoder.train(
        lr = args.lr,
        data_dir = args.data_dir,
        epochs = args.epochs,
        batch_size = args.batch_size,
        eval_interval = args.eval_interval,
    )

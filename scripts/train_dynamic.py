import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import wandb
from torch.utils.data import Dataset, Subset
import numpy as np
from scripts.utilities import parse_argument, set_seed

class TrajectoryDataset(Dataset):
    def __init__(self, data_dict):
        self.obs = data_dict['obs']
        self.actions = data_dict['actions']
        self.time = data_dict['times']
        self.state = data_dict['states']

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.time[idx],
            self.state[idx]
        )

class DynamicModel(nn.Module):
    def __init__(self,
        obs_dim = 8,
        act_dim = 4,
        hidden_dim = 512,
        lr = 1e-3,
        data_dir = 'data.pt',
        epoches = 1000,
        logger = None,
        device = "cuda:0",
        output_dir = "best_dynamic_model.pt"
    ):
        super().__init__()
        input_dim = obs_dim + act_dim + 1
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
            # nn.Tanh()
        )
        self.device = device
        self.model.to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        dataset = torch.load(f'data/{data_dir}')
        self.dataset = {}
        for key, values in dataset.items():
            self.dataset[key] = values
        self.dataset = TrajectoryDataset(self.dataset)
        self.eval_interval = 1
        self.batch_size = 2048
        self.epoches = epoches
        self.output_dir = output_dir
        self.logger = logger
        self.a = torch.tensor([-0.4118, -0.1927], device=self.device)
        self.b = torch.tensor([0.3932, 0.1938], device=self.device)
        self.scale = (self.b - self.a) / 2
        self.shift = (self.a + self.b) / 2


    def forward(self, obs, act, t):
        # obs = obs[:, 2].unsqueeze(1)
        x = torch.cat([obs, act, t.float() / 200], dim=1)
        x = self.model(x)
        # return self.scale * x + self.shift
        return x

    def fit(self):
        best_val_loss = float('inf')
        best_model_state = None


        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_set, val_set = random_split(self.dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        for epoch in tqdm(range(self.epoches)):
            total_train_loss = 0
            for obs, action, time, state in train_loader:
                obs = obs.to(self.device).float()
                action = action.to(self.device).float()
                time = time.to(self.device).float()
                state = state.to(self.device).float()

                pred_state = self.forward(obs, action, time)
                loss = self.loss_fn(pred_state, state)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * obs.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            if self.logger is not None:
                self.logger.log({
                    'dynamic_train_loss' : avg_train_loss,
                }, step=epoch)

            if epoch % self.eval_interval == 0:
                total_val_loss = 0
                with torch.no_grad():
                    for obs, action, time, state in val_loader:
                        obs = obs.to(self.device).float()
                        action = action.to(self.device).float()
                        time = time.to(self.device).float()
                        state = state.to(self.device).float()

                        pred_state = self.forward(obs, action, time)
                        loss = self.loss_fn(pred_state, state)
                        total_val_loss += loss.item() * obs.size(0)

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if self.logger is not None:
                    self.logger.log({
                        "eval/loss": avg_val_loss,
                    }, step=epoch)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict()
                    torch.save(best_model_state, f'output/{self.output_dir}')
                    print(">> Saved best model")
            
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f}")
            
        print("Training finished. Best validation loss:", best_val_loss)


class RnnTrajectoryDataset(Dataset):
    def __init__(self, data_dict, episode_len=40):
        self.episode_len = episode_len
        self.obs = data_dict['obs']
        self.actions = data_dict['actions']
        self.time = data_dict['times']
        self.state = data_dict['states']

        # 總共可以切出幾個完整 episode
        self.num_episodes = self.obs.shape[0] // episode_len

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        start = idx * self.episode_len
        end = (idx + 1) * self.episode_len

        obs_seq = self.obs[start:end]           # shape: [T, obs_dim]
        actions_seq = self.actions[start:end]   # shape: [T, act_dim]
        time_seq = self.time[start:end]         # shape: [T, 1] or [T]
        state_seq = self.state[start:end]       # shape: [T, state_dim]

        return obs_seq, actions_seq, time_seq, state_seq

class RNNModel(nn.Module):
    def __init__(self, obs_dim = 8,
        act_dim = 4,
        output_dim = 2,
        hidden_dim = 32,
        lr = 1e-3,
        data_dir = 'data.pt',
        epoches = 1000,
        logger = None,
        device = "cuda:0",
        output_dir = "best_dynamic_model.pt"
    ):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device
        self.rnn.to(self.device)
        self.fc.to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        dataset = torch.load(f'data/{data_dir}')
        self.dataset = {}
        for key, values in dataset.items():
            self.dataset[key] = values
        self.dataset = RnnTrajectoryDataset(self.dataset)
        self.eval_interval = 1
        self.batch_size = 256
        self.epoches = epoches
        self.output_dir = output_dir
        self.logger = logger
        
    def forward(self, obs, act, time):
        # x: [B, T, D]
        x = torch.cat([obs, act], dim=2)
        rnn_out, _ = self.rnn(x)       
        return self.fc(rnn_out)
    def fit(self):
        best_val_loss = float('inf')
        best_model_state = None


        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_set, val_set = random_split(self.dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        for epoch in tqdm(range(self.epoches)):
            total_train_loss = 0
            for obs, action, time, state in train_loader:
                obs = obs.to(self.device).float()
                action = action.to(self.device).float()
                time = time.to(self.device).float()
                state = state.to(self.device).float()

                pred_state = self.forward(obs, action, time)
                loss = self.loss_fn(pred_state, state)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * obs.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            if self.logger is not None:
                self.logger.log({
                    'dynamic_train_loss' : avg_train_loss,
                }, step=epoch)

            if epoch % self.eval_interval == 0:
                total_val_loss = 0
                with torch.no_grad():
                    for obs, action, time, state in val_loader:
                        obs = obs.to(self.device).float()
                        action = action.to(self.device).float()
                        time = time.to(self.device).float()
                        state = state.to(self.device).float()

                        pred_state = self.forward(obs, action, time)
                        loss = self.loss_fn(pred_state, state)
                        total_val_loss += loss.item() * obs.size(0)
                    

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if self.logger is not None:
                    self.logger.log({
                        "eval/loss": avg_val_loss,
                    }, step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.state_dict()
                    torch.save(best_model_state, f'output/{self.output_dir}')
                    print(">> Saved best model")
            
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f}")
        print("Training finished. Best validation loss:", best_val_loss)

if __name__ == '__main__':
    args = parse_argument()
    set_seed(args.seed)
    wandb.init(
        project='ev_drl',
    )

    if args.method == "mlp":
        dynamic = DynamicModel(data_dir=args.data_dir, logger=wandb, output_dir=args.output_dir)
        dynamic.fit()
    else:
        rnn = RNNModel(data_dir=args.data_dir, logger=wandb, output_dir=args.output_dir)
        rnn.fit()

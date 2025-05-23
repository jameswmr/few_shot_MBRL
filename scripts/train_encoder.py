import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import wandb
from torch.utils.data import Dataset

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
        hidden_dim = 128,
        lr = 1e-3,
        data_dir = 'enc_data.pt',
        epoches = 1000,
        logger = None,
        device = "cuda:2"
    ):
        super().__init__()
        input_dim = trajectory_dim + obs_dim + act_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.device = device
        self.model.to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.dataset = torch.load(f'data/{data_dir}')
        self.dataset = EncoderDataset(self.dataset)
        self.eval_interval = 10
        self.batch_size = 64
        self.epoches = epoches
        self.logger = logger


    def forward(self, traj1, obs, act):
        # print(traj1.shape)
        x = torch.cat([traj1, obs, act], dim=1)
        # print(x.shape)
        return self.model(x)

    def train(self):
        best_val_loss = float('inf')
        best_model_state = None


        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_set, val_set = random_split(self.dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        for epoch in tqdm(range(self.epoches)):
            total_train_loss = 0
            for traj, real_traj, obs, action, env_params in train_loader:
                traj = traj.to(self.device).float()
                real_traj = real_traj.to(self.device).float()
                obs = obs.to(self.device).float()
                action = action.to(self.device).float()
                env_params = env_params.to(self.device).float()

                pred_env_params = self.forward(real_traj[:, :40], obs, action)
                loss = self.loss_fn(pred_env_params, env_params)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * obs.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)


            if epoch % self.eval_interval == 0:
                total_val_loss = 0
                with torch.no_grad():
                    for traj, real_traj, obs, action, env_params in train_loader:
                        traj = traj.to(self.device).float()
                        real_traj = real_traj.to(self.device).float()
                        obs = obs.to(self.device).float()
                        action = action.to(self.device).float()
                        env_params = env_params.to(self.device).float()

                        pred_env_params = self.forward(real_traj[:, :40], obs, action)
                        loss = self.loss_fn(pred_env_params, env_params)
                        total_val_loss += loss.item() * obs.size(0)

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict()
                    torch.save(best_model_state, 'output/best_encoder_traj_20.pt')
                    print(">> Saved best encoder")
            
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f}")
            if self.logger is not None:
                self.logger.log({
                    'encoder_train_loss' : avg_train_loss,
                })
        print("Training finished. Best validation loss:", best_val_loss)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='enc_data.pt')
    args = parser.parse_args()
    # wandb.init(
    #     project='ev_drl',
    # )

    encoder = Encoder(data_dir=args.data_dir,)# logger=wandb)
    encoder.train()

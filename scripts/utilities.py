import random
import numpy as np
import torch
from gym_env.utilities import ENV_CONTEXTS, SAC_CONFIG, PPO_CONFIG
from gym_env.utilities import RandomContextStateWrapper, NormalizeActionSpaceWrapper, SetContextStateWrapper
from stable_baselines3.common.monitor import Monitor
import argparse
def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def make_env_f(env_name, dr):
    def _make_env():
        Env = ENV_CONTEXTS[env_name]["constructor"]
        env = Env(render=False)
        env = RandomContextStateWrapper(env, env_name, dr)
        env = NormalizeActionSpaceWrapper(env)
        return env
    # env = Monitor(env)
    return _make_env

def make_env(env_name, dr, sample_train=False):
    Env = ENV_CONTEXTS[env_name]['constructor']
    env = Env(render=False)
    print(f'{sample_train=}')
    if sample_train:
        env = SetContextStateWrapper(env, env_name, dr)
    else:
        env = RandomContextStateWrapper(env, env_name, dr)
    env = NormalizeActionSpaceWrapper(env)
    return env

def generate_sample_train(data_dir, N=100):
    data = torch.load(f'data/{data_dir}')
    idxs = np.random.choice(len(data['obs']), size=N, replace=False)
    init_pos = []
    actions = []
    obs = []
    env_params = []
    for idx in idxs: 
        env_params.append({"dynamic@floor1_table_collision@friction_sliding": data['obs'][idx][2],
                    "dynamic@floor2_table_collision@friction_sliding": data['obs'][idx][3],
                    "dynamic@knob_g0@damping_ratio": data['obs'][idx][4],
                    "dynamic@x_left_wall_g0@damping_ratio": data['obs'][idx][5],
                    "dynamic@x_right_wall_g0@damping_ratio": data['obs'][idx][6],
                    "dynamic@block_g0@damping_ratio": data['obs'][idx][7],})
        init_pos.append(data['init_pos'][idx//40][:2])
        actions.append(data['actions'][idx])
        obs.append(data['obs'][idx])
    return init_pos, actions, obs, env_params

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="push_one", help="Environment name"
    )
    parser.add_argument(
        "--policy_name", type=str, choices=["ppo", "sac"], help="Policy name"
    )
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=150000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--save_freq", type=int, default=150, help="Checkpoint interval"
    )
    parser.add_argument(
        "--pre_trained_model",type=str, default=None, help="Pre-trained model path"
    )
    parser.add_argument(
        "--dr", type=bool, default=False, help="Enable domain randomization"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=100, help="Data Collection for n episodes"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Set device"
    )
    parser.add_argument(
        "--final_state", type=bool, default=False, help="Only collect final state"
    )
    parser.add_argument(
        "--dym_dir", type=str, default="best_dynamic_model.pt", help="Set load model directory"
    )
    parser.add_argument(
        "--enc_dir", type=str, default="epoch_1960.ckpt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data.pt", help="Set output path"
    )
    parser.add_argument('--data_dir', type=str, default='data.pt')
    parser.add_argument(
        '--method', type=str, default="mlp"
    )
    parser.add_argument(
        '--sample_train', type=bool, default=False, help="Sample eval data from training data"
    )

    # train_encoder
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-interval', type=int, default=20)
    return parser.parse_args()
    
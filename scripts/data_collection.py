import numpy as np
import torch
from tqdm import tqdm
from scripts.utilities import set_seed, make_env, parse_argument


def main(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)

    # env = AsyncVectorEnv([make_env(args.env_name, args.dr) for _ in range(args.n_envs)])
    env = make_env(args.env_name, args.dr)
    # print(env)
    obss = []
    actionss = []
    statess = []
    times = []
    init_pos = []
    for _ in tqdm(range(args.n_episodes)):
        # env_params = {"dynamic@floor1_table_collision@friction_sliding": np.random.uniform(-1, 1),
        #         "dynamic@floor2_table_collision@friction_sliding": -1,
        #         "dynamic@knob_g0@damping_ratio": 1,
        #         "dynamic@x_left_wall_g0@damping_ratio": 1,
        #         "dynamic@x_right_wall_g0@damping_ratio": 1,
        #         "dynamic@block_g0@damping_ratio": 1,}
        # obs = env.env.reset(env_params, [-0.147, 0.036])
        obs = env.reset()
        # print(env.cup_init_pos, obs)
        # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(args.n_envs)]), device=args.device)
        actions = env.action_space.sample()
        nxt_obs, reward, done, info = env.step(actions)
        trajectories = env.get_state_traj(info)
        init_pos.append(env.cup_init_pos)
        # print(trajectories)
        if args.final_state:
            obss.append(obs)
            actionss.append(actions)
            times.append(39)
            statess.append(trajectories[-1])
        else: 
            for t in range(40):
                obss.append(obs)
                actionss.append(actions)
                times.append(t)
                statess.append(trajectories[t])
    obss_tensor = torch.tensor(np.array(obss), dtype=torch.float32)      
    actionss_tensor = torch.tensor(np.array(actionss), dtype=torch.float32) 
    times_tensor = torch.tensor(np.array(times), dtype=torch.float32).unsqueeze(1)  
    statess_tensor = torch.tensor(np.array(statess), dtype=torch.float32) 
    init_pos_tensor = torch.tensor(np.array(init_pos), dtype=torch.float32)
    data = {
        'obs': obss_tensor,
        'actions': actionss_tensor,
        'times': times_tensor,
        'states': statess_tensor,
        'init_pos': init_pos_tensor
    }
    torch.save(data, f'data/{args.output_dir}')





    

if __name__ == "__main__":
    args = parse_argument()
    main(args)
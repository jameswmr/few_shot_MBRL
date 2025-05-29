# push_one_task.py
import numpy as np
import torch
import gym
from gym import spaces
from few_shot_MBRL.gym_env.env_push_one import PusherOneSingleAction

class PushOneTask(gym.Env):                          
    def __init__(self,
                 config=None,               
                 sim_device='cpu',
                 graphics_device_id=0,
                 headless=True):
        metadata = {"render.modes": []}
        self.env = PusherOneSingleAction(render=False)
        
        self.action_space      = self.env.action_space                
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(2,), dtype=np.float32)
        
        self.priv_keys = [
            "cup_init_pos_x",                
            "cup_init_pos_y",                
            "cup_radius",                    
            "cup_height",                    
            "cup_mass",                      
            "cup_com_offset_x",              
            "cup_com_offset_y",              
            "cup_com_offset_z",              
            "cup_friction_sliding",          
        ]
        self.priv_info_dim = 9

    def _extract_priv(self):
        return torch.from_numpy(self.env.env_params)


    def reset(self):
        obs_raw = self.env.reset()
        if isinstance(obs_raw, tuple):
            obs_raw = obs_raw[0]

        priv = self._extract_priv()
        return {"obs": torch.tensor(obs_raw, dtype=torch.float32),
                "priv_info": priv}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        priv = self._extract_priv()       
        info["env_params"] = priv.numpy() 
        return ({"obs": torch.tensor(obs, dtype=torch.float32),
                 "priv_info": priv},
                torch.tensor([rew], dtype=torch.float32),
                torch.tensor(done),
                info)

    def render(self, **kwargs):
        self.env.render()
    def close(self):
        self.env.close()
    def seed(self, seed=None):
        self.env.seed(seed)
        return [seed]

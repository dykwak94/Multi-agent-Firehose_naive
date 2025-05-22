# rl_experiment_multiagent.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from firehose.models import PaddedNatureCNN
from gym_env_multi import MultiAgentFireEnv

# Optional: adjust these as you like
MAP_NAME = "Sub20x20"
N_STEPS = 5_000_000
AGENTS = ["helicopter", "drone", "groundcrew"]

def make_env():
    return MultiAgentFireEnv(
        fire_map=MAP_NAME,
        observation_type="forest_rgb",
        action_type="flat"
    )

# --- Multi-agent wrapper for SB3 ---
class SB3MultiAgentWrapper:
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env, agents, train_agent="helicopter"):
        self.env = env
        self.agents = agents
        self.train_agent = train_agent
        self.observation_space = env.observation_spaces[train_agent]
        self.action_space = env.action_spaces[train_agent]

    def reset(self):
        obs_dict = self.env.reset()
        return obs_dict[self.train_agent]  # Only one agent's observation

    def step(self, action):
        # Provide action only for the training agent
        action_dict = {agent: 0 for agent in self.agents}  # dummy actions
        action_dict[self.train_agent] = action
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        obs = obs_dict[self.train_agent]
        reward = reward_dict[self.train_agent]
        done = done_dict[self.train_agent]
        info = info_dict[self.train_agent]
        return obs, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        pass

# ---- Training ----

if __name__ == "__main__":
    # SB3 expects a single-agent Gym env, so we use a wrapper.
    # For true multi-policy, see MARL libs.
    env = SB3MultiAgentWrapper(make_env(), AGENTS)
    # For compatibility with SB3, wrap in DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_logs",
        policy_kwargs={"features_extractor_class": PaddedNatureCNN},
    )

    model.learn(total_timesteps=N_STEPS)
    model.save("multiagent_firehose_ppo_cnn.zip")
    env.close()

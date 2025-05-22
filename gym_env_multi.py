# gym_env_multi.py
from gym import Env, spaces
import numpy as np
from cell2fire.gym_env import FireEnv
from firehose.rewards_multi import HelicopterReward, DroneReward, GroundCrewReward


AGENT_IDS = ["helicopter", "drone", "groundcrew"]

def get_agent_observation(env, agent):
    obs = env.get_observation()
    return obs  # All agents get the full map for now (same shape)

class MultiAgentFireEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, **fireenv_kwargs):
        # Create shared FireEnv (simulates the forest, fire, etc.)
        self.env = FireEnv(**fireenv_kwargs)
        self.reward_funcs = {
            "helicopter": HelicopterReward(self.env),
            "drone": DroneReward(self.env),
            "groundcrew": GroundCrewReward(self.env),
        }
        self.agents = AGENT_IDS
        # For now, same obs/action for all, but can be changed per agent later
        self.observation_spaces = {agent: self.env.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space for agent in self.agents}
        self.last_obs = None
        self.last_rewards = None

    def reset(self, **kwargs):
        obs_dict = {agent: get_agent_observation(self.env, agent) for agent in self.agents}
        # For now, all agents get the same observation (can be agent-specific later)
        self.last_obs = obs_dict
        self.last_rewards = {agent: 0.0 for agent in self.agents}
        return obs_dict

    def step(self, action_dict):
        """
        action_dict: {agent_id: action}
        For now, we just let each agent act in turn (could combine into one joint action later).
        """
        # For now, let the "helicopter" act, then "drone", then "groundcrew" in sequence.
        # (This is a simple scheme; later we can support simultaneous/more complex schemes.)
        total_reward = 0
        done = False
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}

        # For each agent, apply their action and accumulate effect.
        # In this simple version, each step is a "macro-step": all agents act before env advances.
        for agent in self.agents:
            obs, reward, done, info = self.env.step(action_dict[agent])
            obs_dict[agent] = obs
            # Use agent-specific reward
            reward_dict[agent] = self.reward_funcs[agent]()

            done_dict[agent] = done
            info_dict[agent] = info
            if done:
                break  # Episode ended

        self.last_obs = obs_dict
        self.last_rewards = reward_dict
        # Gym compatibility: return (obs, reward, done, info)
        # If you want PettingZoo compatibility, this could be easily wrapped!
        return obs_dict, reward_dict, done_dict, info_dict

    def render(self, mode="human", **kwargs):
        return self.env.render(mode=mode, **kwargs)

    def close(self):
        self.env.close()

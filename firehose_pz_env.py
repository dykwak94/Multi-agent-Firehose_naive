# firehose_pz_env.py
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from cell2fire.gym_env import FireEnv
from firehose.rewards_multi import HelicopterReward, DroneReward, GroundCrewReward

AGENT_NAMES = ["helicopter", "drone", "groundcrew"]

class FirehosePettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "firehose_v0"}

    def __init__(self, **fireenv_kwargs):
        self.base_env = FireEnv(**fireenv_kwargs)
        self.agents = AGENT_NAMES.copy()
        self.possible_agents = AGENT_NAMES.copy()

        obs_space = self.base_env.observation_space
        act_space = self.base_env.action_space

        # Print type for debugging
        print(f"obs_space type for agent '{self.agents[0]}':", type(obs_space))
        assert isinstance(obs_space, spaces.Space), "obs_space must be a gymnasium.spaces.Space"
        
        self.observation_spaces = {agent: spaces.Box(low=0, high=255, shape=(20, 20, 3), dtype=np.float32) for agent in self.agents}
        self.action_spaces = {agent: act_space for agent in self.agents}

        self.reward_funcs = {
            "helicopter": HelicopterReward(self.base_env),
            "drone": DroneReward(self.base_env),
            "groundcrew": GroundCrewReward(self.base_env),
        }

    def reset(self, *, seed=None, options=None):
        obs = self.base_env.reset()
        obs = obs.astype(np.float32)
        self.agents = self.possible_agents.copy()
        obs_dict = {agent: obs for agent in self.agents}
        info_dict = {agent: {} for agent in self.agents}
        return obs_dict, info_dict

    def step(self, actions):
        first_agent = self.agents[0]
        scalar_action = actions[first_agent]
        # Step all agents
        # Step the base env ONCE, then assign to all agents
        # (If your env is synchronous, this is usually correct for ParallelEnv)
        
        obs, reward, done, info = self.base_env.step(scalar_action)
        obs = obs.astype(np.float32)
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in self.agents:
            observations[agent] = obs
            rewards[agent] = self.reward_funcs[agent]().astype(np.float32) if agent in self.reward_funcs else 0.0
            terminations[agent] = done  # or use your logic for agent-done
            truncations[agent] = False
            infos[agent] = info

        # RLlib/PettingZoo wants __all__ key for multi-agent compatibility
        terminations["__all__"] = any(terminations[agent] for agent in self.agents)
        truncations["__all__"] = any(truncations[agent] for agent in self.agents)

        if terminations["__all__"] or truncations["__all__"]:
            # Return "done" for all agents but keep in dict for this step
            self.agents = []

        return observations, rewards, terminations, truncations, infos


    def render(self):
        return self.base_env.render(mode="human")

    def close(self):
        self.base_env.close()

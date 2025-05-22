# train_firehose_rllib.py
import ray
import os
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from firehose_pz_env import FirehosePettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

def env_creator(config):
    return ParallelPettingZooEnv(FirehosePettingZooEnv(
        fire_map="Sub20x20",
        observation_type="forest_rgb",
        action_type="flat"
    ))

register_env("firehose_pz", env_creator)

obs_space = FirehosePettingZooEnv(fire_map="Sub20x20", observation_type="forest_rgb", action_type="flat").observation_spaces["helicopter"]
act_space = FirehosePettingZooEnv(fire_map="Sub20x20", observation_type="forest_rgb", action_type="flat").action_spaces["helicopter"]

policy_dict = {
    agent: (None, obs_space, act_space, {}) for agent in ["helicopter", "drone", "groundcrew"]
}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    logs_dir = os.path.abspath("./firehose_rllib_logs")
    config = (
    PPOConfig()
    .environment("firehose_pz")
    .env_runners(num_env_runners=1)
    .multi_agent(
        policies=policy_dict,
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .rl_module(
        model_config={
            "conv_filters": [
                [16, [3, 3], 1],
                [32, [3, 3], 1],
                [64, [3, 3], 1],
            ],
            # You can add more settings here as needed
        }
    )
    .training(
        train_batch_size=200,
        # other training hyperparams
    )
    .resources(num_gpus=0)
    .debugging(log_level="INFO")
)
tune.run(
    "PPO",
    stop={"episodes_total": 1000},
    config=config.to_dict(),
    checkpoint_at_end=True,
    storage_path=f"file://{logs_dir}",
)

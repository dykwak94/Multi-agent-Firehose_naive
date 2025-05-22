# test_firehose_pz.py
from firehose_pz_env import FirehosePettingZooEnv

def main():
    env = FirehosePettingZooEnv(
        fire_map="Sub20x20",
        observation_type="forest_rgb",
        action_type="flat"
    )
    obs, _ = env.reset()
    print("Initial observations:")
    for agent, o in obs.items():
        
        print(f"{agent}: {o.shape}")
    for step in range(5):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"Step {step+1} rewards: {rewards}")
        if not env.agents:
            print("Episode ended!")
            break
    env.close()

if __name__ == "__main__":
    main()

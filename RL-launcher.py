from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.A3C import Agent


def main():
    # env_name = 'CartPole-v1'
    env_name = 'pipeline'
    agent = Agent(env_name)
    agent.train()


if __name__ == "__main__":
    main()

import argparse
import os
import json
import random

import numpy as np
from tqdm import tqdm
from app.greedy_summarizer import GreedySummarizer
from rl.A3C_2_actors.intrinsic_curiosity_model import IntrinsicCuriosityForwardModel
from rl.A3C_2_actors.operation_actor import OperationActor
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from rl.A3C_2_actors.set_actor import SetActor
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str,
                    default="scattered-ccur-0.75-lstm-5-alr-3e-05-clr-3e-05-03082021_182550")

args = parser.parse_args()


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


# model_names = {
#     "LO": "spotify-lo",
#     "HI": "spotify-hi",
#     "BL": "spotify-bl",
#     "DC": "spotify-dc",
#     "IC": "spotify-ic",
#     "DORA": "spotify-dora",
# }

model_names = {
    # "LO": "spotify-lo",
    # "HI": "spotify-hi",
    # "BL": "spotify-trad-bl",
    # "DC": "spotify-dc",
    "IC": "spotify-trad-ic",
    # "DORA": "spotify-dora",
}
# model_names = {
#     # "LO": "sdss-trad-constant-0.45-0.45-0.1-3-11172021_101914",
#     # "HI": "sdss-constant-0.1-0.1-0.8-3-11172021_101617",
#     # "BL": "sdss-constant-0.333-0.333-0.334-3-11172021_101558",
#     "DC": "sdss-trad-decreasing_gamma-0.0005-0.0005-0.999-3-11172021_101831",
#     # "IC": "sdss-trad-increasing_gamma-0.5-0.5-0.0-3-11172021_101819",
#     # "DORA": "sdss-None-0.333-0.333-0.334-3-11172021_101514"
# }

# model_names = {
#     "LO": "sdss-constant-0.45-0.45-0.1-3-11172021_101628",
#     "HI": "sdss-constant-0.1-0.1-0.8-3-11172021_101617",
#     "BL": "sdss-constant-0.333-0.333-0.334-3-11172021_101558",
#     "DC": "sdss-decreasing_gamma-0.0005-0.0005-0.999-3-11172021_101543",
#     "IC": "sdss-increasing_gamma-0.5-0.5-0.0-3-11172021_101525",
#     "DORA": "sdss-None-0.333-0.333-0.334-3-11172021_101514"
# }


class AgentRunner:
    def __init__(self, mode, curiosity_weight, model_path, model_name, pipeline=None, columns=None, result_sets=None, steps=50, args=None):
        data_folder = "./app/data/"

        self.episode_steps = steps
        self.agent_name = args.name
        self.steps = 3  # args.lstm_steps
        if columns == None:
            columns = ["tracks.popularity", "tracks.acousticness", "tracks.danceability",
                       "tracks.duration_ms", "tracks.energy", "tracks.instrumentalness",
                       "tracks.liveness", "tracks.loudness", "tracks.speechiness", "tracks.tempo", "tracks.valence", ]
        if pipeline == None:
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", ["tracks"], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns,
                id_column="tracks.track_id")
        else:
            self.pipeline = pipeline

        # self.pipeline = PipelineWithPrecalculatedSets(
        #     "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"],
        #     id_column="galaxies.objID")

        if 'trad' in model_name:
            operators = ["by_facet", "by_superset"]
        else:
            operators = ["by_facet", "by_superset",
                         "by_neighbors", "by_distribution"]
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=args.target_set, mode=args.mode, episode_steps=self.episode_steps, operators=operators, utility_weights=args.utility_weights)
        if result_sets == None:
            greedy_summarizer = GreedySummarizer(self.env.pipeline)
            uniformity_threshold = 2 if self.pipeline.database_name == "sdss" else 2
            self.startup_sets = greedy_summarizer.get_summary(
                10, uniformity_threshold, 10)
        else:
            self.startup_sets = result_sets
        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim

        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.set_actor = SetActor(
            self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=f"{model_path}/{curiosity_weight}/set_actor")
        self.operation_actor = OperationActor(
            self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=f"{model_path}/{curiosity_weight}/operation_actor")
        if os.path.exists(f"{model_path}/{curiosity_weight}/set_op_counters.json"):
            with open(f"{model_path}/{curiosity_weight}/set_op_counters.json") as f:
                self.set_op_counters = json.load(f)

        self.counter_curiosity_factor = 100/250

    def run(self, times=5):
        results = []
        for i in range(times):
            print(f"---------------------    RUN: {i}")
            done = False
            set_action_steps = [[-1] * self.set_state_dim] * self.steps
            operation_action_steps = [
                [-1] * self.operation_state_dim] * self.steps
            set_state = self.env.reset(initial_sets=self.startup_sets)
            curiosity_rewards = []
            for step_counter in tqdm(range(self.episode_steps)):
                probs = self.set_actor.model.predict(
                    np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                probs = self.env.fix_possible_set_action_probs(probs[0])
                if all(np.isnan(x) for x in probs):
                    set_action = 0
                else:
                    set_action = np.random.choice(
                        self.set_action_dim, p=probs)

                operation_state = self.env.get_operation_state(set_action)
                operation_action_steps.pop(0)
                operation_action_steps.append(operation_state)
                probs = self.operation_actor.model.predict(
                    np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                probs = self.env.fix_possible_operation_action_probs(
                    set_action, probs[0])
                if np.isnan(probs[0]):
                    operation_action = self.env.get_random_operation(
                        set_action)
                else:
                    operation_action = np.random.choice(
                        self.operation_action_dim, p=probs)

                next_set_state, reward, done, set_op_pair = self.env.step(
                    set_action, operation_action)
                if set_op_pair in self.set_op_counters:
                    self.set_op_counters[set_op_pair] += 1
                else:
                    self.set_op_counters[set_op_pair] = 1

                op_counter = self.set_op_counters[set_op_pair]
                next_set_action_steps = set_action_steps.copy()
                next_set_action_steps.pop(0)
                next_set_action_steps.append(next_set_state)
                curiosity_rewards.append({
                    "curiosity_reward": self.counter_curiosity_factor/op_counter
                })
            for i in range(len(self.env.episode_info)):
                self.env.episode_info[i].update(curiosity_rewards[i])
            results.append(self.env.episode_info)
        return results


# mode = "Scattered"
# # curiosity_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
# curiosity_weights = ["best_utility"]
# res = {}
# for variant, model_name in model_names.items():
#     model_path = "./saved_models/"+model_name
#     for curiosity_weight in curiosity_weights:
#         print(
#             f"---------------------           LOADING: {mode} {curiosity_weight}")
#         with open(f"{model_path}/info.json") as f:
#             items = json.load(f)
#             for key in items.keys():
#                 setattr(args, key, items[key])

#             res[f"{mode}-{curiosity_weight}"] = AgentRunner(
#                 mode, curiosity_weight, model_path, scale_values=False).run(5)
#     with open(f"./{variant}.json", 'w') as f:
#         json.dump(res, f, indent=1, default=np_encoder)

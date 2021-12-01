import argparse
import os
import json
import random

import numpy as np
from tqdm import tqdm
from rl.A3C_2_actors.intrinsic_curiosity_model import IntrinsicCuriosityForwardModel
from rl.A3C_2_actors.operation_actor import OperationActor
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from rl.A3C_2_actors.set_actor import SetActor
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

data_folder = "./app/data/"


class AgentRunner:
    def __init__(self, pipeline):

        self.episode_steps = 200
        self.steps = 3
        # columns = ["dm-authors.seniority", "dm-authors.nb_publi",
        #            "dm-authors.pub_rate", "dm-authors.first_author_por", "dm-authors.avg_coauthor",
        #            "dm-authors.CIKM", "dm-authors.ICDE", "dm-authors.ICWSM", "dm-authors.IEEE", "dm-authors.RecSys",
        #            "dm-authors.SIGIR", "dm-authors.SIGMOD", "dm-authors.VLDB", "dm-authors.WSDM", "dm-authors.WWW"]
        # self.pipeline = PipelineWithPrecalculatedSets(
        #     "dm-authors", ["dm-authors"], data_folder=data_folder, discrete_categories_count=5, min_set_size=10, exploration_columns=columns, id_column="dm-authors.author_id")

        self.pipeline = pipeline

        # self.pipeline = PipelineWithPrecalculatedSets(
        #     "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=None, mode="scattered", episode_steps=self.episode_steps, operators=["by_facet", "by_superset", "by_neighbors", "by_distribution"])

        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n

        self.counter_curiosity_factor = 100/250

    def run(self, times=5):
        results = []
        for i in range(times):
            print(f"---------------------    RUN: {i}")
            self.env.reset()
            for step_counter in tqdm(range(self.episode_steps)):
                probs = self.env.fix_possible_set_action_probs(
                    [0.1]*self.set_action_dim)
                if all(np.isnan(x) for x in probs):
                    set_action = 0
                else:
                    set_action = np.random.choice(
                        self.set_action_dim, p=probs)

                probs = self.env.fix_possible_operation_action_probs(
                    set_action, [1/self.env.action_manager.operation_action_dim]*self.env.action_manager.operation_action_dim)

                operation_action = np.random.choice(
                    self.operation_action_dim, p=probs)

                self.env.step(
                    set_action, operation_action)

            results.append(self.env.episode_info)
        return results


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


# ds_name = "tracks_23000"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
# res = {}
# res[f"random"] = AgentRunner(pipeline).run(10)
# with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
#     json.dump(res, f, indent=1, default=np_encoder)

# ds_name = "tracks_115000"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
# res = {}
# res[f"random"] = AgentRunner(pipeline).run(10)
# with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
#     json.dump(res, f, indent=1, default=np_encoder)

ds_name = "tracks_5b"
columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
    "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=5, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
res = {}
res[f"random"] = AgentRunner(pipeline).run(10)
with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
    json.dump(res, f, indent=1, default=np_encoder)

# ds_name = "tracks_20b"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=20, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
# res = {}
# res[f"random"] = AgentRunner(pipeline).run(10)
# with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
#     json.dump(res, f, indent=1, default=np_encoder)

# ds_name = "tracks_7c"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness", f"{ds_name}.liveness"]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
# res = {}
# res[f"random"] = AgentRunner(pipeline).run(10)
# with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
#     json.dump(res, f, indent=1, default=np_encoder)


# ds_name = "tracks_3c"
# columns = [f"{ds_name}.popularity",
#            f"{ds_name}.acousticness", f"{ds_name}.danceability"]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False, id_column=f"{ds_name}.track_id")
# res = {}
# res[f"random"] = AgentRunner(pipeline).run(10)
# with open(f"./app/data/spotify/{ds_name}_index/random.json", 'w') as f:
#     json.dump(res, f, indent=1, default=np_encoder)

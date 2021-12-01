import json
from tqdm import tqdm
import numpy as np
import argparse
from app import summary_evaluator
from app.utility_manager import UtilityManager
from collections import Counter, OrderedDict
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from app.summary_evaluator import SummaryEvaluator
from app.greedy_summarizer import GreedySummarizer

parser = argparse.ArgumentParser()
parser.add_argument('--utility_mode', type=str, default="constant")
parser.add_argument('--utility_weights', nargs='+',
                    type=float, default=[0.45, 0.45, 0.1])
args = parser.parse_args()


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


data_folder = "./app/data/"
# pipeline = PipelineWithPrecalculatedSets(
#     "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10,
#     exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r",
#                          "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"],
#     id_column="galaxies.objID")
# columns = ["tracks.popularity", "tracks.acousticness", "tracks.danceability",
#            "tracks.duration_ms", "tracks.energy", "tracks.instrumentalness",
#            "tracks.liveness", "tracks.loudness", "tracks.speechiness", "tracks.tempo", "tracks.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", ["tracks"], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns,
#     id_column="tracks.track_id")

# greedy_summarizer = GreedySummarizer(pipeline)


class Top1Sum:
    def __init__(self, variant, utility_mode, utility_weights, columns=None, pipeline=None, start_set="all_data", trad=False, swap_threshold=2, result_sets=None, steps=50):
        print()
        print(utility_mode)
        print(utility_weights)
        self.episode_steps = steps

        # self.pipeline = PipelineWithPrecalculatedSets(
        #     "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10,
        #     exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r",
        #                          "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"],
        #     id_column="galaxies.objID")
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

        if result_sets == None:
            greedy_summarizer = GreedySummarizer(self.pipeline)
            result_sets = greedy_summarizer.get_summary(
                10, swap_threshold, 10)
        if not trad:
            operators = ["by_facet", "by_superset",
                         "by_neighbors", "by_distribution"]
        else:
            operators = ["by_facet", "by_superset"]
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=None, mode="scattered", episode_steps=self.episode_steps, operators=operators,
            utility_mode=utility_mode, utility_weights=utility_weights)

        self.env.reset(result_sets)
        # if start_set == "all_data":
        #     self.env.datasets = [self.pipeline.get_dataset()]
        # elif start_set == "smallest_set":
        #     self.env.datasets = self.pipeline.get_groups_as_datasets(
        #         [len(self.pipeline.groups)-1])
        # elif start_set == "median_size_set":
        #     self.env.datasets = self.pipeline.get_groups_as_datasets(
        #         [round(len(self.pipeline.groups)/2)])
        previous_operations = []
        for i in tqdm(range(self.episode_steps)):
            set_ids = set([set.set_id for set in self.env.datasets])
            if set_ids == None:
                set_ids = set()
            future_scores = self.env.utility_manager.get_future_scores(
                self.env.datasets, self.pipeline, self.env.sets_viewed, set_ids, self.env.utility_weights, previous_operations)
            future_scores = [
                x for x in future_scores if x["operation"] in operators]
            if future_scores[0]["setId"] == None:
                set_index = 0
            else:
                set_index = self.env.datasets.index(next(
                    (x for x in self.env.datasets if x.set_id == future_scores[0]["setId"]), None))

            action = future_scores[0]["operation"]
            attribute = ""
            if future_scores[0]["attribute"] != "":
                attribute += f"{self.pipeline.initial_collection_names[0]}.{future_scores[0]['attribute']}"
                action += f"-&-{self.pipeline.initial_collection_names[0]}.{future_scores[0]['attribute']}"
            self.env.step(
                set_index, self.env.action_manager.set_action_types.index(action))
            previous_operations.append(
                f'{future_scores[0]["operation"]}-{attribute}-{future_scores[0]["setId"]}')
        mode = "trad" if trad else "all"
        if swap_threshold == 2:
            with open(f"./runs-data/top1sum/spotify/{mode}/{variant}.json", 'w') as f:
                json.dump(self.env.episode_info, f,
                          indent=1, default=np_encoder)
        else:
            with open(f"./runs-data/top1sum/spotify/{mode}/{variant}_{swap_threshold}.json", 'w') as f:
                json.dump(self.env.episode_info, f,
                          indent=1, default=np_encoder)


# Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
#         "all_data", trad=False, swap_threshold=1.5)

# Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
#         "all_data", trad=False, swap_threshold=2.5)
# Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
#         "all_data", trad=False, swap_threshold=3)
# Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
#         "all_data", trad=False, swap_threshold=3.5)
# Top1Sum("LO", "constant", [0.45, 0.45, 0.1],
#         "all_data", trad=False, swap_threshold=3)
# Top1Sum("BL", "constant", [0.333, 0.333, 0.334],
#         "all_data", trad=False, swap_threshold=3)
# Top1Sum("DC", "decreasing_gamma", [
#         0.0005, 0.0005, 0.999], "all_data", trad=False, swap_threshold=3)
# Top1Sum("IC", "increasing_gamma", [0.5, 0.5, 0],
#         "all_data", trad=False, swap_threshold=3)

# for threshold in np.arange(4, 4.5, 0.5):
#     result_sets = greedy_summarizer.get_summary(
#         10, threshold, 10)
#     Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
#             "all_data", trad=False, swap_threshold=threshold, result_sets=result_sets)
#     Top1Sum("LO", "constant", [0.45, 0.45, 0.1],
#             "all_data", trad=False, swap_threshold=threshold, result_sets=result_sets)
#     Top1Sum("BL", "constant", [0.333, 0.333, 0.334],
#             "all_data", trad=False, swap_threshold=threshold, result_sets=result_sets)
#     Top1Sum("DC", "decreasing_gamma", [
#             0.0005, 0.0005, 0.999], "all_data", trad=False, swap_threshold=threshold, result_sets=result_sets)
#     Top1Sum("IC", "increasing_gamma", [0.5, 0.5, 0],
#             "all_data", trad=False, swap_threshold=threshold, result_sets=result_sets)


# # Top1Sum("constant", [0.1, 0.1, 0.8], "smallest_set")
# # Top1Sum("constant", [0.4, 0.4, 0.2], "smallest_set")
# # Top1Sum("constant", [0.45, 0.45, 0.1], "smallest_set")
# Top1Sum("constant", [0.333, 0.333, 0.334], "smallest_set")
# Top1Sum("decreasing_gamma", [0.0005, 0.0005, 0.999], "smallest_set")
# Top1Sum("increasing_gamma", [0.5, 0.5, 0], "smallest_set")

# # Top1Sum("constant", [0.1, 0.1, 0.8], "median_size_set")
# # Top1Sum("constant", [0.4, 0.4, 0.2], "median_size_set")
# # Top1Sum("constant", [0.45, 0.45, 0.1], "median_size_set")
# Top1Sum("constant", [0.333, 0.333, 0.334], "median_size_set")
# Top1Sum("decreasing_gamma", [0.0005, 0.0005, 0.999], "median_size_set")
# Top1Sum("increasing_gamma", [0.5, 0.5, 0], "median_size_set")

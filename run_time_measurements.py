from timeit import default_timer as timer
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
from run_top1sum import Top1Sum
from run_agent import AgentRunner

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str,
                    default="scattered-ccur-0.75-lstm-5-alr-3e-05-clr-3e-05-03082021_182550")

args = parser.parse_args()

length = 30
data_folder = "./app/data/"

# ds_name = "tracks_3c"
db_name = "spotify"
ds_name = "tracks"
columns = ["tracks.popularity", "tracks.acousticness", "tracks.danceability",
           "tracks.duration_ms", "tracks.energy", "tracks.instrumentalness",
           "tracks.liveness", "tracks.loudness", "tracks.speechiness", "tracks.tempo", "tracks.valence", ]

pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
    "spotify", ["tracks"], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns,
    id_column="tracks.track_id")
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=5, min_set_size=20, exploration_columns=columns, scale_values=True, id_column=f"{ds_name}.track_id")

greedy_summarizer = GreedySummarizer(pipeline)

result_sets = greedy_summarizer.get_summary(
    10, 2, 5)

start = timer()
Top1Sum("HI", "constant", [0.1, 0.1, 0.8],
        pipeline=pipeline, columns=columns, trad=False, swap_threshold=2, result_sets=result_sets, steps=length)
end = timer()

step_time = (end - start)/length

print(f"top1sum step time {step_time}")
model_path = f"./saved_models/{db_name}-hi"
with open(f"{model_path}/info.json") as f:
    items = json.load(f)
    for key in items.keys():
        setattr(args, key, items[key])
    runner = AgentRunner(
        "Scattered", "best_utility", model_path, f"{db_name}-hi", pipeline=pipeline, columns=columns, result_sets=result_sets, steps=length, args=args)

start = timer()
runner.run(1)
end = timer()

step_time = (end - start)/length
print(f"rlsum step time {step_time}")

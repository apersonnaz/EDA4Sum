import json
from tqdm import tqdm
import numpy as np
import argparse
from app.utility_manager import UtilityManager
from scipy.spatial.distance import cityblock
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from app.summary_evaluator import SummaryEvaluator

data_folder = "./app/data/"

min_set_size = 10
min_uniformity_target = 1.5
result_set_count = 10


def get_min_distance(groups):
    min_distance = -1
    for index1, dataset1 in enumerate(groups):
        for dataset2 in groups[index1+1: len(groups)]:
            distance = cityblock(
                dataset1["means_vector"], dataset2["means_vector"])
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
    return min_distance


utility_manager = UtilityManager()
for i in range(1024):
    if utility_manager.zscore(utility_manager.boxcox(i, lmbda=utility_manager.uniformity_lambda), utility_manager.uniformity_mean, utility_manager.uniformity_std) >= min_uniformity_target:
        min_uniformity = i
        break

print(f'Min uniformity: {min_uniformity}')

pipeline = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])


groups = pipeline.groups
groups = groups[(groups.member_count >= min_set_size) & (groups.uniformity >= min_uniformity)
                ].sort_values(by="uniformity", ascending=False).reset_index().to_dict('records')

candidates = groups[0:result_set_count]
candidates_distance = get_min_distance(candidates)
print(f'start dist: {candidates_distance}')
for group in tqdm(groups):
    if not group in candidates:
        for i in range(result_set_count):
            new_candidates = candidates.copy()
            new_candidates[i] = group
            new_candidates_distance = get_min_distance(new_candidates)
            if new_candidates_distance > candidates_distance:
                candidates = new_candidates
                candidates_distance = new_candidates_distance
                print(f'new dist: {candidates_distance}')
                break
datasets = pipeline.get_groups_as_datasets([x["id"] for x in candidates])
for dataset in datasets:
    print(str(dataset.predicate))

print([x["id"] for x in candidates])
summary_evaluator = SummaryEvaluator(pipeline)
summary_evaluator.evaluate_sets(datasets)
print(summary_evaluator.get_galaxy_class_score())
print(summary_evaluator.get_inverted_entropy_score())

res = [
 {
  "input_set_index": None,
  "input_set_size": None,
  "input_set_id": None,
  "operator": None,
  "parameter": None,
  "output_set_count": None,
  "output_set_average_size": None,
  "reward": None,
  "sets_viewed": None,
  "sets_reviewed": None,
  "uniformity": 0,
  "diversity": 0,
  "novelty": 0,
  "utility": 0,
  "utility_weights": None,
  "extrinsic_reward": 0,
  "inverted_entropy_score": summary_evaluator.get_inverted_entropy_score(),
            "galaxy_class_score": summary_evaluator.get_galaxy_class_score(),
            "galaxy_class_mean_distance": summary_evaluator.get_galaxy_class_mean_distance(),
            "class_distance_found": summary_evaluator.get_found_class_distance_count(1),
            "class_distance_found_0.1": summary_evaluator.get_found_class_distance_count(0.1),
            "class_distance_found_0.06": summary_evaluator.get_found_class_distance_count(0.06),
            "class_distance_found_0.05": summary_evaluator.get_found_class_distance_count(0.05),
            "class_distance_found_0.04": summary_evaluator.get_found_class_distance_count(0.04),
            "class_distance_found_0.03": summary_evaluator.get_found_class_distance_count(0.03),
            "class_distance_found_0.02": summary_evaluator.get_found_class_distance_count(0.02),
            "class_score_found": summary_evaluator.get_found_class_score_count(0),
            "class_score_found_3": summary_evaluator.get_found_class_score_count(3),
            "class_score_found_4": summary_evaluator.get_found_class_score_count(4),
            "class_score_found_5": summary_evaluator.get_found_class_score_count(5),
            "class_score_found_8": summary_evaluator.get_found_class_score_count(8),
            "class_score_found_12": summary_evaluator.get_found_class_score_count(12),
 }]


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

with open(f"./runs-data/greedysum.json", 'w') as f:
            json.dump(res, f, indent=1, default=np_encoder)
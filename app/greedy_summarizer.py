import json
from tqdm import tqdm
import numpy as np
import argparse
from scipy.spatial.distance import cityblock
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets


data_folder = "./app/data/"

min_set_size = 10
min_uniformity_target = 2
result_set_count = 10


class GreedySummarizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.utility_manager = pipeline.utility_manager

    def get_min_distance(self, groups):
        min_distance = -1000
        for index1, dataset1 in enumerate(groups):
            for dataset2 in groups[index1+1: len(groups)]:
                distance = cityblock(
                    dataset1["means_vector"], dataset2["means_vector"])
                if min_distance == -1000 or distance < min_distance:
                    min_distance = distance
        return min_distance

    def get_rax_min_uniformity(self, min_uniformity_target):
        for i in range(1024):
            if self.utility_manager.zscore(self.utility_manager.boxcox(i, lmbda=self.utility_manager.uniformity_lambda), self.utility_manager.uniformity_mean, self.utility_manager.uniformity_std) >= min_uniformity_target:
                return i

    def get_summary(self, min_set_size=10, min_uniformity_target=2, result_set_count=10):
        min_uniformity = self.get_rax_min_uniformity(min_uniformity_target)
        groups = self.pipeline.groups
        groups = groups[(groups.member_count >= min_set_size) & (groups.uniformity >= min_uniformity)
                        ].sort_values(by="uniformity", ascending=False).reset_index().to_dict('records')
        selected_sets = groups[0:result_set_count]
        current_distance = self.get_min_distance(selected_sets)
        print(f'start dist: {current_distance}')
        for group in tqdm(groups):
            if not group in selected_sets:
                best_candidates_distance = current_distance
                for i in range(result_set_count):
                    new_candidates = selected_sets.copy()
                    new_candidates[i] = group
                    new_candidates_distance = self.get_min_distance(
                        new_candidates)
                    if new_candidates_distance > best_candidates_distance:
                        best_candidates = new_candidates
                        best_candidates_distance = new_candidates_distance
                if best_candidates_distance > current_distance:
                    selected_sets = best_candidates
                    current_distance = best_candidates_distance
                    print(f'new dist: {current_distance}')
        return self.pipeline.get_groups_as_datasets([x["id"] for x in selected_sets])

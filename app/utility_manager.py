import json
import math
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cityblock


class UtilityManager:
    def __init__(self, scaling_file, scale_values=True):
        self.scale_values = scale_values
        if self.scale_values:
            with open(scaling_file) as f:
                self.scaling_data = json.load(f)
            self.uniformity_lambda = self.scaling_data["uniformity"]["lambda"]
            self.uniformity_mean = self.scaling_data["uniformity"]["mean"]
            self.uniformity_std = self.scaling_data["uniformity"]["std"]
            self.diversity_lambda = self.scaling_data["diversity"]["lambda"]
            self.diversity_mean = self.scaling_data["diversity"]["mean"]
            self.diversity_std = self.scaling_data["diversity"]["std"]
            self.novelty_lambda = self.scaling_data["novelty"]["lambda"]
            self.novelty_mean = self.scaling_data["novelty"]["mean"]
            self.novelty_std = self.scaling_data["novelty"]["std"]

    def boxcox(self, value, lmbda):
        if lmbda != 0:
            return (value**lmbda - 1) / lmbda
        else:
            return math.log(value)

    def zscore(self, value, mean, std):
        return (value - mean)/std

    def normalize_uniformity(self, value):
        return self.zscore(self.boxcox(
            value, lmbda=self.uniformity_lambda), self.uniformity_mean, self.uniformity_std)

    def get_uniformity_scores(self, datasets, pipeline):
        set_uniformities = []
        for dataset in datasets:
            if self.scale_values:
                uniformity = self.boxcox(
                    dataset.uniformity, lmbda=self.uniformity_lambda)
                uniformity = self.zscore(
                    uniformity, self.uniformity_mean, self.uniformity_std)
                set_uniformities.append(uniformity)
            else:
                set_uniformities.append(dataset.uniformity)

        return min(set_uniformities) if len(datasets) > 1 else 0, set_uniformities

    def get_min_distance(self, datasets, pipeline):
        if len(datasets) <= 1:
            return 0
        min_distance = -1
        for index1, dataset1 in enumerate(datasets):
            for dataset2 in datasets[index1+1: len(datasets)]:
                distance = cityblock(
                    dataset1.means_vector, dataset2.means_vector)/len(pipeline.exploration_columns)
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
        if self.scale_values:
            return self.zscore(self.boxcox(min_distance, lmbda=self.diversity_lambda), self.diversity_mean, self.diversity_std)
        else:
            return min_distance

    def get_novelty_scores_and_utility_weights(self, datasets, seen_sets, pipeline, decreasing_gamma=False, utility_weights=None):
        summary_set_ids = set(map(lambda x: int(x.set_id), datasets))
        if len(datasets) <= 1:
            return 0, seen_sets | summary_set_ids, utility_weights
        summary_score = len(summary_set_ids - seen_sets)/len(summary_set_ids)
        if self.scale_values:
            summary_score = self.zscore(self.boxcox(
                summary_score, lmbda=self.novelty_lambda), self.novelty_mean, self.novelty_std)
        seen_sets = seen_sets | summary_set_ids
        if decreasing_gamma:
            gamma = math.pow((len(seen_sets | summary_set_ids) /
                              (len(pipeline.groups)/1000))-1, 2)
            if gamma < 0:
                gamma = 0
            alpha_beta = (1 - gamma) / 2
            # alpha_beta = (1 - gamma)
        else:
            alpha_beta = math.pow((len(seen_sets | summary_set_ids) /
                                   (len(pipeline.groups)/1000))-1, 2)/2
            # alpha_beta = math.pow((len(seen_sets | summary_set_ids) /
            #                        (len(pipeline.groups)/1000))-1, 2)
            if alpha_beta < 0:
                alpha_beta = 0
            gamma = 1 - 2 * alpha_beta
            # gamma = 1 - alpha_beta
        return summary_score, seen_sets | summary_set_ids, [alpha_beta, alpha_beta, gamma]

    def compute_utility(self, weights, uniformity, diversity, novelty):
        # return weights[0] * uniformity * diversity + weights[2] * novelty
        return weights[0] * uniformity + weights[1] * diversity + weights[2] * novelty
        # return weights[0] * uniformity + weights[1] * diversity * math.exp(uniformity) + weights[2] * novelty

    def get_future_scores(self, sets, pipeline, seen_sets, previous_dataset_ids, utility_weights, previous_operations):
        operations = []
        for index, dataset in enumerate(sets):
            if dataset.set_id != None and dataset.set_id >= 0 or len(dataset.data) == len(pipeline.initial_collection):
                predicate_attributes = dataset.predicate.get_attributes()
                for attribute in pipeline.exploration_columns:
                    if attribute in predicate_attributes:
                        if len(dataset.predicate.components) <= 1:
                            continue
                        operation = "by_neighbors"
                        operation_identifier = f"{operation}-{attribute}-{dataset.set_id}"
                        if not operation_identifier in previous_operations:
                            resulting_sets = pipeline.by_neighbors(
                                dataset=dataset, attributes=[attribute], return_data=False)
                    else:
                        operation = "by_facet"
                        operation_identifier = f"{operation}-{attribute}-{dataset.set_id}"
                        if not operation_identifier in previous_operations:
                            resulting_sets = pipeline.by_facet(
                                dataset=dataset, attributes=[attribute], number_of_groups=10, return_data=False)

                    if not operation_identifier in previous_operations:
                        resulting_sets = [d for d in resulting_sets if d.set_id !=
                                          None and d.set_id >= 0]
                        if len(resulting_sets) > 0 and set(map(lambda x: x.set_id, resulting_sets)) != previous_dataset_ids:
                            summary_uniformity, sets_uniformity_scores = self.get_uniformity_scores(
                                resulting_sets, pipeline)
                            summary_distance = self.get_min_distance(
                                resulting_sets, pipeline)
                            summary_novelty, final_predicates, new_utility_weights = self.get_novelty_scores_and_utility_weights(
                                resulting_sets, seen_sets, pipeline, utility_weights=utility_weights)
                            operations.append({
                                "setId": int(dataset.set_id) if dataset.set_id != None else None,
                                "operation": operation,
                                "attribute": attribute.replace(f"{pipeline.initial_collection_names[0]}.", ""),
                                "uniformity": summary_uniformity,
                                "novelty": summary_novelty,
                                "distance": summary_distance,
                                "utility": self.compute_utility(utility_weights, summary_uniformity,
                                                                summary_distance, summary_novelty)
                            })
                if len(dataset.predicate.components) > 1:
                    operation_identifier = f"by_distribution--{dataset.set_id}"
                    if not operation_identifier in previous_operations:
                        resulting_sets = pipeline.by_distribution(
                            dataset=dataset)
                        resulting_sets = [d for d in resulting_sets if d.set_id !=
                                          None and d.set_id >= 0]
                        if len(resulting_sets) > 0 and set(map(lambda x: x.set_id, resulting_sets)) != previous_dataset_ids:
                            summary_uniformity, sets_uniformity_scores = self.get_uniformity_scores(
                                resulting_sets, pipeline)
                            summary_distance = self.get_min_distance(
                                resulting_sets, pipeline)
                            summary_novelty, final_predicates, new_utility_weights = self.get_novelty_scores_and_utility_weights(
                                resulting_sets, seen_sets, pipeline, utility_weights=utility_weights)
                            operations.append({
                                "setId": int(dataset.set_id) if dataset.set_id != None else None,
                                "operation": "by_distribution",
                                "attribute": "",
                                "uniformity": summary_uniformity,
                                "novelty": summary_novelty,
                                "distance": summary_distance,
                                "utility": self.compute_utility(utility_weights, summary_uniformity,
                                                                summary_distance, summary_novelty)
                            })

                    operation_identifier = f"by_superset--{dataset.set_id}"
                    if not operation_identifier in previous_operations:
                        resulting_sets = pipeline.by_superset(
                            dataset=dataset, return_data=False)
                        resulting_sets = [d for d in resulting_sets if d.set_id !=
                                          None and d.set_id >= 0]
                        if len(resulting_sets) > 0 and set(map(lambda x: x.set_id, resulting_sets)) != previous_dataset_ids:
                            summary_uniformity, sets_uniformity_scores = self.get_uniformity_scores(
                                resulting_sets, pipeline)
                            summary_distance = self.get_min_distance(
                                resulting_sets, pipeline)
                            summary_novelty, final_predicates, new_utility_weights = self.get_novelty_scores_and_utility_weights(
                                resulting_sets, seen_sets, pipeline, utility_weights=utility_weights)
                            operations.append({
                                "setId": int(dataset.set_id) if dataset.set_id != None else None,
                                "operation": "by_superset",
                                "attribute": "",
                                "uniformity": summary_uniformity,
                                "novelty": summary_novelty,
                                "distance": summary_distance,
                                "utility": self.compute_utility(utility_weights, summary_uniformity,
                                                                summary_distance, summary_novelty)
                            })
        return sorted(operations, key=lambda k: k['utility'], reverse=True)

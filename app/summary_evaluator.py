import json
from scipy.spatial.distance import cityblock
from scipy.stats import entropy
import math


class SummaryEvaluator:
    def __init__(self, pipeline, scale_values=True, galaxy_class_scores=None) -> None:
        self.pipeline = pipeline
        self.utility_manager = pipeline.utility_manager
        if self.pipeline.database_name == 'sdss':
            with open("galaxy_classes_mean_vectors.json") as f:
                self.galaxy_classes = json.load(f)
            self.galaxy_class_scores = dict.fromkeys(
                list(self.galaxy_classes.keys()), None) if galaxy_class_scores == None else galaxy_class_scores
            self.best_galaxy_class_score_sets = {}
            self.galaxy_class_distances = dict.fromkeys(
                list(self.galaxy_classes.keys()), None)
            self.best_galaxy_class_distance_sets = {}
        elif self.pipeline.database_name == 'spotify':
            self.genre_ratios = dict.fromkeys(
                self.pipeline.initial_collection[f"{self.pipeline.initial_collection_names[0]}.genre"].unique(), 0)

        min_non_null_entropy = self.pipeline.groups[self.pipeline.groups.entropy != 0].sort_values(
            by="entropy").iloc[0].entropy
        max_inverted_entropy = 1/min_non_null_entropy
        self.max_inverted_entropy = max_inverted_entropy * 0.1

        self.inverted_entropy_score = 0
        self.seen_sets = []
        self.scale_values = scale_values
        if self.scale_values:
            self.min_normalized_uniformity = self.utility_manager.normalize_uniformity(
                self.pipeline.groups.uniformity.min())
            self.uniformity_offset = abs(
                self.min_normalized_uniformity) if self.min_normalized_uniformity < 0 else 0
        # entropies = []
        # for attribute in pipeline.exploration_columns:
        #     value_counts = self.pipeline.initial_collection[attribute].value_counts()
        #     entropies.append(entropy(value_counts))
        # max_entropy = sum(entropies)

    def evaluate_sets(self, datasets):
        for dataset in datasets:
            dataset_scores = {}
            dataset_distances = {}

            set_uniformity = self.utility_manager.normalize_uniformity(
                dataset.uniformity) + self.uniformity_offset if self.scale_values else dataset.uniformity
            if self.pipeline.database_name == 'sdss':
                for g_class, galaxy_class_score in self.galaxy_class_scores.items():
                    set_distance = cityblock(
                        dataset.means_vector, self.galaxy_classes[g_class])/len(dataset.means_vector)
                    set_score = set_uniformity/set_distance
                    if galaxy_class_score == None or set_score > galaxy_class_score:
                        self.galaxy_class_scores[g_class] = set_score
                        self.best_galaxy_class_score_sets[g_class] = dataset
                    if self.galaxy_class_distances[g_class] == None or set_distance < self.galaxy_class_distances[g_class]:
                        self.galaxy_class_distances[g_class] = set_distance
                        self.best_galaxy_class_distance_sets[g_class] = dataset
            elif self.pipeline.database_name == 'spotify':
                for genre, genre_best_ratio in self.genre_ratios.items():
                    genre_ratio = len(
                        dataset.data[dataset.data[f"{self.pipeline.initial_collection_names[0]}.genre"] == genre])/len(dataset.data)
                    if genre_ratio > genre_best_ratio:

                        print(
                            f"new best for {genre} {genre_best_ratio} => {genre_ratio}")
                        self.genre_ratios[genre] = genre_ratio
            #     dataset_scores[g_class] = set_score
            #     dataset_distances[g_class] = set_distance
            # max_score_class = max(dataset_scores, key=dataset_scores.get)
            # if dataset_scores[max_score_class] > self.galaxy_class_scores[max_score_class]:
            #     self.galaxy_class_scores[max_score_class] = dataset_scores[max_score_class]
            #     self.best_galaxy_class_score_sets[max_score_class] = dataset

            # min_distance_class = min(dataset_distances, key=dataset_scores.get)
            # if dataset_distances[min_distance_class] < self.galaxy_class_distances[min_distance_class]:
            #     self.galaxy_class_distances[min_distance_class] = dataset_distances[min_distance_class]
            #     self.best_galaxy_class_distance_sets[min_distance_class] = dataset

            if not dataset.set_id in self.seen_sets:
                self.seen_sets.append(dataset.set_id)
                if dataset.entropy == 0:
                    self.inverted_entropy_score += self.max_inverted_entropy
                else:
                    self.inverted_entropy_score += 1 / dataset.entropy

    def get_evaluation_scores(self):
        if self.pipeline.database_name == 'sdss':
            return {
                "galaxy_class_score": self.get_mean_score(),
                "class_score_found_12": self.get_found_class_score_count(12),
                "class_score_found_15": self.get_found_class_score_count(15),
                "class_score_found_18": self.get_found_class_score_count(18),
                "class_score_found_21": self.get_found_class_score_count(21),
            }
            # "galaxy_class_mean_distance": self.summary_evaluator.get_galaxy_class_mean_distance(),
            # "class_distance_found": self.summary_evaluator.get_found_class_distance_count(1),
            # "class_distance_found_0.1": self.summary_evaluator.get_found_class_distance_count(0.1),
            # "class_distance_found_0.06": self.summary_evaluator.get_found_class_distance_count(0.06),
            # "class_distance_found_0.05": self.summary_evaluator.get_found_class_distance_count(0.05),
            # "class_distance_found_0.04": self.summary_evaluator.get_found_class_distance_count(0.04),
            # "class_distance_found_0.03": self.summary_evaluator.get_found_class_distance_count(0.03),
            # "class_distance_found_0.02": self.summary_evaluator.get_found_class_distance_count(0.02),
            # "class_score_found": self.summary_evaluator.get_found_class_score_count(0),
            # "class_score_found_3": self.summary_evaluator.get_found_class_score_count(3),
            # "class_score_found_4": self.summary_evaluator.get_found_class_score_count(4),
            # "class_score_found_5": self.summary_evaluator.get_found_class_score_count(5),
            # "class_score_found_8": self.summary_evaluator.get_found_class_score_count(8),
        elif self.pipeline.database_name == 'spotify':
            return {
                "genre_mean_ratio": self.get_mean_score(),
                "found_genre_10": self.get_found_genre_count(0.1),
                "found_genre_20": self.get_found_genre_count(0.2),
                "found_genre_30": self.get_found_genre_count(0.3),
                "found_genre_40": self.get_found_genre_count(0.4),
                "found_genre_50": self.get_found_genre_count(0.5),
                "found_genre_60": self.get_found_genre_count(0.6),
                "found_genre_70": self.get_found_genre_count(0.7),
                "found_genre_80": self.get_found_genre_count(0.8),
                "found_genre_90": self.get_found_genre_count(0.9),
                "found_genre_100": self.get_found_genre_count(1),
            }

    def get_mean_score(self):
        if self.pipeline.database_name == 'sdss':
            return sum(self.galaxy_class_scores.values())/len(self.galaxy_class_scores)
        elif self.pipeline.database_name == 'spotify':
            return sum(self.genre_ratios.values())/len(self.genre_ratios)

    def get_found_genre_count(self, threshold):
        counter = 0
        for ratio in self.genre_ratios.values():
            if ratio >= threshold:
                counter += 1
        return counter

    def get_inverted_entropy_score(self):
        return self.inverted_entropy_score

    def get_galaxy_class_mean_distance(self):
        if self.pipeline.database_name == 'sdss':
            return sum(self.galaxy_class_distances.values())/len(self.galaxy_class_distances)
        else:
            return 0

    def get_found_class_distance_count(self, threshold):
        if self.pipeline.database_name == 'sdss':
            counter = 0
            for distance in self.galaxy_class_distances.values():
                if distance < threshold:
                    counter += 1
            return counter
        else:
            return 0

    def get_found_class_score_count(self, threshold):
        if self.pipeline.database_name == 'sdss':
            counter = 0
            for distance in self.galaxy_class_scores.values():
                if distance > threshold:
                    counter += 1
            return counter
        else:
            return 0

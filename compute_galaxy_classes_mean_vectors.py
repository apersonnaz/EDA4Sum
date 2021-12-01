import os
import json
import numpy as np
from app.utility_manager import UtilityManager
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


data_folder = "./app/data/"
pipeline = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])

galaxy_classes = {}
original_attributes = list(
    map(lambda x: x+"_original", list(pipeline.ordered_dimensions)))
original_attributes += [
    x for x in pipeline.exploration_columns if not x in pipeline.ordered_dimensions]
for attribute in original_attributes:
    mean = pipeline.initial_collection[attribute].mean()
    std = pipeline.initial_collection[attribute].std()
    pipeline.initial_collection[attribute] = (
        pipeline.initial_collection[attribute] - mean) / std
    # pipeline.initial_collection[attribute] = (pipeline.initial_collection[attribute] - pipeline.initial_collection[attribute].min()) / (
    #     pipeline.initial_collection[attribute].max() - pipeline.initial_collection[attribute].min())
for filename in os.listdir("rl/targets"):
    with open("rl/targets/"+filename) as f:
        ids = json.load(f)
        galaxies = pipeline.initial_collection[pipeline.initial_collection["galaxies.objID"].isin(
            ids)]
        means_vector = []
        for attribute in original_attributes:
            if attribute.replace("_original", "") in pipeline.ordered_dimensions:
                mean = galaxies[attribute].mean()
                means_vector.append(mean)
        galaxy_classes[filename.replace(
            '.json', '')] = means_vector

with open(f"./galaxy_classes_mean_vectors.json", 'w') as f:
    json.dump(galaxy_classes, f, indent=1, default=np_encoder)

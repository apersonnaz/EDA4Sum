from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor
import statistics
from app.pipelines.pipeline_precalculated_sets import \
    PipelineWithPrecalculatedSets
from scipy.stats import entropy


def process_computing(start, stop, pipeline: PipelineWithPrecalculatedSets):
    results = {}
    original_attributes = list(
        map(lambda x: x+"_original", list(pipeline.ordered_dimensions)))
    original_attributes += [
        x for x in pipeline.exploration_columns if not x in pipeline.ordered_dimensions]
    # attributes_min_max = {}
    # for attribute in original_attributes:
    #     attributes_min_max[attribute] = {'min': pipeline.initial_collection[attribute].min(
    #     ), 'max': pipeline.initial_collection[attribute].max()}
    for i in tqdm(range(start, stop)):
        dataset = pipeline.get_groups_as_datasets([i])[0]
        # if len(dataset.predicate.components) < len(pipeline.exploration_columns):
        #     attributes_for_variance = set(pipeline.exploration_columns).difference(
        #         set(map(lambda x: x.attribute, dataset.predicate.components)))
        # else:
        # attributes_for_variance = original_attributes
        # attributes_in_desc = set(
        #     map(lambda x: x.attribute, dataset.predicate.components))
        # original_data = dataset.data[original_attributes]
        # variances = original_data.var(axis=0)
        entropies = []
        stds = []
        means_vector = []
        for attribute in original_attributes:
            if attribute.replace("_original", "") in pipeline.ordered_dimensions:
                mean = dataset.data[attribute].mean()
                means_vector.append(mean)
                stds.append(dataset.data[attribute].std())

        for attribute in pipeline.exploration_columns:
            value_counts = dataset.data[attribute].value_counts()
            entropies.append(entropy(value_counts))

        # results[i] = {'uniformity': 1 /
        #               (sum(entropies)/len(entropies)), 'means_vector': means_vector, 'entropy': sum(entropies)}
        results[i] = {'uniformity': 1 /
                      (sum(stds)/len(stds)), 'means_vector': means_vector, 'entropy': sum(entropies), 'mean_std': sum(stds)/len(stds)}
    return results


data_folder = "./app/data/"


def compute_uniformities(ds_name, pipeline):

    group_file = f"{data_folder}spotify/{ds_name}_index/groups.csv"
    # columns = ["dm-authors.seniority", "dm-authors.nb_publi",
    #            "dm-authors.pub_rate", "dm-authors.first_author_por", "dm-authors.avg_coauthor",
    #            "dm-authors.CIKM", "dm-authors.ICDE", "dm-authors.ICWSM", "dm-authors.IEEE", "dm-authors.RecSys",
    #            "dm-authors.SIGIR", "dm-authors.SIGMOD", "dm-authors.VLDB", "dm-authors.WSDM", "dm-authors.WWW"]
    # pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
    #     "dm-authors", ["dm-authors"], data_folder=data_folder, discrete_categories_count=5, min_set_size=10, exploration_columns=columns)
    # group_file = data_folder+"dm-authors/dm-authors_index/groups.csv"
    # pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
    #     "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
    # group_file = data_folder+"sdss/galaxies_index/groups.csv"
    original_attributes = list(
        map(lambda x: x+"_original", list(pipeline.ordered_dimensions)))
    for attribute in original_attributes:
        mean = pipeline.initial_collection[attribute].mean()
        std = pipeline.initial_collection[attribute].std()
        pipeline.initial_collection[attribute] = (
            pipeline.initial_collection[attribute] - mean) / std
        # pipeline.initial_collection[attribute] = (pipeline.initial_collection[attribute] - pipeline.initial_collection[attribute].min()) / (
        #     pipeline.initial_collection[attribute].max() - pipeline.initial_collection[attribute].min())
    #pipeline.groups = pipeline.groups.iloc[0:503]
    uniformities = []
    datasets = []
    futures = list()
    workers_count = 40
    with ProcessPoolExecutor(max_workers=workers_count) as executor:
        process_group_count = len(pipeline.groups) // workers_count
        for i in range(workers_count):
            start = i * process_group_count
            if i < workers_count - 1:
                stop = (i+1)*process_group_count
            else:
                stop = len(pipeline.groups)
            futures.append(executor.submit(process_computing,
                                           start=start,
                                           stop=stop,
                                           pipeline=pipeline))

        results_dict = {}
        for future in futures:
            results_dict.update(future.result())

        uniformities = []
        means_vectors = []
        entropies = []
        mean_stds = []
        for i in range(len(pipeline.groups)):
            uniformities.append(results_dict[i]["uniformity"])
            means_vectors.append(results_dict[i]["means_vector"])
            entropies.append(results_dict[i]["entropy"])
            mean_stds.append(results_dict[i]["mean_std"])

        # min_uniformity = min(uniformities)
        # max_uniformity = max(uniformities)
        # normalized_uniformities = list(map(lambda x: (
        #     x - min_uniformity)/(max_uniformity-min_uniformity), uniformities))
        # uniformities_mean = statistics.mean(uniformities)
        # uniformities_std = statistics.pstdev(uniformities)
        # z_score_uniformities = list(
        #     map(lambda x: (x - uniformities_mean)/uniformities_std, uniformities))
        groups = pipeline.groups
        groups["uniformity"] = uniformities
        # groups["uniformity"] = normalized_uniformities
        # groups["z_score_uniformity"] = z_score_uniformities
        groups["means_vector"] = means_vectors
        groups["entropy"] = entropies
        groups["mean_std"] = mean_stds
        groups.to_csv(group_file)


# ds_name = "tracks_23000"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False)
# compute_uniformities(ds_name, pipeline)

# ds_name = "tracks_115000"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False)
# compute_uniformities(ds_name, pipeline)

ds_name = "tracks_5b"
columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
    "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=5, min_set_size=20, exploration_columns=columns, scale_values=False)
compute_uniformities(ds_name, pipeline)

# ds_name = "tracks_20b"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
#            f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=20, min_set_size=20, exploration_columns=columns, scale_values=False)
# compute_uniformities(ds_name, pipeline)

# ds_name = "tracks_7c"
# columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
#            f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness", f"{ds_name}.liveness"]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False)
# compute_uniformities(ds_name, pipeline)


# ds_name = "tracks_3c"
# columns = [f"{ds_name}.popularity",
#            f"{ds_name}.acousticness", f"{ds_name}.danceability"]
# pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
#     "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=False)
# compute_uniformities(ds_name, pipeline)

# for i in tqdm(range(len(pipeline.groups))):
#     # if i % 500 == 0:
#     #     if i+500 < len(pipeline.groups):
#     #         datasets = pipeline.get_groups_as_datasets(list(range(i, i+500)))
#     #     else:
#     #         datasets = pipeline.get_groups_as_datasets(
#     #             list(range(i, len(pipeline.groups))))
#     # dataset = datasets[i % 500]
#     dataset = pipeline.get_groups_as_datasets([i])[0]
#     attributes_in_desc = set(
#         map(lambda x: x.attribute, dataset.predicate.components))
#     original_data = dataset.data[list(
#         map(lambda x: x+"_original", attributes_in_desc))]
#     variances = original_data.var(axis=0)
#     uniformities.append(1/variances.max())

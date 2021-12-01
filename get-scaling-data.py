import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


# files = ["scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10082021_151020.json",
# "scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10082021_151158.json",
# "scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10082021_152041.json",
# "scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10082021_152049.json",
# "scattered-ccur-0.75-lstm-3-alr-3e-05-clr-3e-05-10082021_152055.json",
# "scattered-ccur-0.25-lstm-3-alr-3e-05-clr-3e-05-10082021_152100.json",
# "scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10082021_152103.json",
# "scattered-no-curiosity-3-alr-3e-05-clr-3e-05-10112021_152951.json"]
for ds_name in ["tracks_5b"]:
    files = ["random.json"]
    unique_data = {}
    for filename in files:
        with open(f"app/data/spotify/{ds_name}_index/{filename}") as f:
            data = json.load(f)
        for run in data["random"]:
            # for run in data['Scattered-best_reward']:
            for step in run:
                if not f"{step['input_set_id']}-{step['operator']}-{step['parameter']}" in unique_data and step["uniformity"] != 0:
                    unique_data[f"{step['input_set_id']}-{step['operator']}-{step['parameter']}"] = step

    print(len(unique_data))

    uniformity = [d["uniformity"] for d in unique_data.values()]
    diversity = [d["diversity"] for d in unique_data.values()]
    novelty = [d["novelty"] for d in unique_data.values()]

    measures = ["uniformity", "diversity", "novelty"]

    scaling_data = {}
    for measure in measures:
        measure_data = [d[measure] for d in unique_data.values()]
        fig = plt.figure(figsize=(8, 8))
        sns.distplot(measure_data, label="original")
        bc_measure, lambdaa = scipy.stats.boxcox(
            [u + 0.0000000001 for u in measure_data])
        fig = plt.figure(figsize=(8, 8))
        sns.distplot(bc_measure, label="boxcox")
        zs_measure = scipy.stats.zscore(bc_measure)
        fig = plt.figure(figsize=(8, 8))
        sns.distplot(zs_measure, label="zscore")

        scaling_data[measure] = {}
        scaling_data[measure]["lambda"] = lambdaa
        scaling_data[measure]["mean"] = np.mean(bc_measure)
        scaling_data[measure]["std"] = np.std(bc_measure)
        print(scaling_data[measure])
        fig.legend()
        # plt.show()

    print(scaling_data)
    with open(f"app/data/spotify/{ds_name}_index/scaling_data.json", 'w') as f:
        json.dump(scaling_data, f, indent=1, default=np_encoder)

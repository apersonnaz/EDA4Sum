import argparse
import json
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

database = 'sdss'
operators = ["by_facet", "by_superset",
                         "by_neighbors", "by_distribution"]
operator_modes = ["all", "trad"]
variants = ["IC", "LO", "BL", "HI", "DC"]
markers = ["o", "v", "s", "*", "p", "P",
           "h", "X", "D", "+", 2, 3, '$/$', '$]$']
runs_data = {
    "top1sum": {
        "all": {

        },
        "trad": {

        }
    },
    "rlsum": {
        "all": {

        },
        "trad": {

        }
    }
}
# plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "bold"
plt.rc("axes", labelweight="bold")
for variant in variants:
    for mode in operator_modes:
        if os.path.isfile(f"./runs-data/rlsum/{database}/{mode}/{variant}.json"):
            with open(f"./runs-data/rlsum/{database}/{mode}/{variant}.json") as f:
                data = json.load(f)
                cumulated_utilities = []
                cumulated_uniformities = []
                cumulated_diversities = []
                cumulated_novelties = []
                scores = []
                operation_counters = []
                for run in data["Scattered-best_utility"]:
                    cumulated_utility = np.cumsum([x["utility"] for x in run])
                    cumulated_utilities.append(cumulated_utility)
                    cumulated_uniformities.append(
                        np.cumsum([x["uniformity"] for x in run]))
                    cumulated_diversities.append(
                        np.cumsum([x["diversity"] for x in run]))
                    cumulated_novelties.append(
                        np.cumsum([x["novelty"] for x in run]))
                    if database == 'sdss':
                        scores.append([x["class_score_found_15"] for x in run])
                    else:
                        scores.append([x["found_genre_50"] for x in run])
                    operation_counters.append(
                        Counter([d["operator"] for d in run]))
                mean_counters = {}
                std_counters = {}
                for operator in operators:
                    mean_counters[operator] = np.mean(
                        [x[operator] for x in operation_counters])
                    std_counters[operator] = np.std(
                        [x[operator] for x in operation_counters])
                runs_data["rlsum"][mode][variant] = {
                    "utility": {
                        "mean": [np.mean(k) for k in zip(*cumulated_utilities)],
                        "std": [np.std(k) for k in zip(*cumulated_utilities)]
                    },
                    "diversity": {
                        "mean": [np.mean(k) for k in zip(*cumulated_diversities)],
                        "std": [np.std(k) for k in zip(*cumulated_diversities)]
                    },
                    "novelty": {
                        "mean": [np.mean(k) for k in zip(*cumulated_novelties)],
                        "std": [np.std(k) for k in zip(*cumulated_novelties)]
                    },
                    "uniformity": {
                        "mean": [np.mean(k) for k in zip(*cumulated_uniformities)],
                        "std": [np.std(k) for k in zip(*cumulated_uniformities)]
                    },
                    "score": {
                        "mean": [np.mean(k) for k in zip(*scores)],
                        "std": [np.std(k) for k in zip(*scores)]
                    },
                    "operation_counters": {
                        "mean": mean_counters,
                        "std": std_counters
                    }
                }

for variant in variants:
    for mode in operator_modes:
        if os.path.isfile(f"./runs-data/top1sum/{database}/{mode}/{variant}.json"):
            with open(f"./runs-data/top1sum/{database}/{mode}/{variant}.json") as f:
                data = json.load(f)
                cumulated_utility = np.cumsum([x["utility"] for x in data])
                runs_data["top1sum"][mode][variant] = {
                    "utility": cumulated_utility,
                    "uniformity": np.cumsum([x["uniformity"] for x in data]),
                    "diversity": np.cumsum([x["diversity"] for x in data]),
                    "novelty": np.cumsum([x["novelty"] for x in data]),
                    "operation_counters": Counter([d["operator"] for d in data])
                }
                if database == 'sdss':
                    runs_data["top1sum"][mode][variant]["score"] = [
                        x["class_score_found_15"] for x in data]
                else:
                    runs_data["top1sum"][mode][variant]["score"] = [
                        x["found_genre_50"] for x in data]


print(runs_data)
figsize = (6, 2.7)
histosize = (14, 2.1)
t = range(50)

# UTILITY
# x = np.arange(len(list(processed_data.keys())))
run_labels = ["Top1Sum_"+x for x in variants]+["RLSum_"+x for x in variants]
x = np.arange(len(run_labels))
utilities = [runs_data["top1sum"]["all"][x]["utility"][-1]
             for x in variants] + [runs_data["rlsum"]["all"][x]["utility"]["mean"][-1]
                                   for x in variants]

width = 0.8  # the width of the bars

fig, ax = plt.subplots(figsize=histosize)
rects1 = ax.bar(x, utilities, width,
                label='utility', edgecolor='black')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('cumulated values')
ax.set_xticks(x)
ax.set_xticklabels(run_labels)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(
    f'graphs/{database}-utility_histogram.pdf', bbox_inches='tight')
# plt.show()


uniformities = [runs_data["top1sum"]["all"][x]["uniformity"][-1]
                for x in variants]+[runs_data["rlsum"]["all"][x]["uniformity"]["mean"][-1]
                                    for x in variants]
diversities = [runs_data["top1sum"]["all"][x]["diversity"][-1]
               for x in variants]+[runs_data["rlsum"]["all"][x]["diversity"]["mean"][-1]
                                   for x in variants]
novelties = [runs_data["top1sum"]["all"][x]["novelty"][-1]
             for x in variants]+[runs_data["rlsum"]["all"][x]["novelty"]["mean"][-1]
                                 for x in variants]

width = 0.3
fig, ax = plt.subplots(figsize=histosize)
rects2 = ax.bar(x - width, uniformities, width,
                label='uniformity', edgecolor='black', color="red", hatch='\\\\')
rects3 = ax.bar(x, diversities, width,
                label='diversity', edgecolor='black', color="green", hatch='o')
rects4 = ax.bar(x + width, novelties,
                width, label='novelty', edgecolor='black', color="orange", hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('cumulated values')
ax.set_xticks(x)
ax.set_xticklabels(run_labels)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(
    f'graphs/{database}-utility_details_histogram.pdf', bbox_inches='tight')
# plt.show()


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
metric = "utility"
vals = OrderedDict()
for key in runs_data["rlsum"]["all"].keys():
    ax.plot(t, runs_data["rlsum"]["all"][key][metric]
            ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
    vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key][metric]["mean"][-1]
for key in runs_data["top1sum"]["all"].keys():
    ax.plot(t, runs_data["top1sum"]["all"][key]
            [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
    vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key][metric][-1]
# if database == 'sdss':
#     ax.plot(t, runs_data["rlsum"]["trad"]["IC"][metric]
#             ["mean"], label=f"RLSum_IC_2OP", marker=markers[10], markevery=5)
#     ax.plot(t, runs_data["top1sum"]["trad"]["LO"][metric],
#             label=f"Top1Sum_LO_2OP", marker=markers[11], markevery=5)
# else:
#     ax.plot(t, runs_data["rlsum"]["trad"]["HI"][metric]
#             ["mean"], label=f"RLSum_HI_2OP", marker=markers[10], markevery=5)
#     ax.plot(t, runs_data["top1sum"]["trad"]["IC"][metric],
#             label=f"Top1Sum_IC_2OP", marker=markers[11], markevery=5)
ax.set_xlabel('pipeline length')
ax.set_ylabel(f'cumulated {metric}')
handles, labels = plt.gca().get_legend_handles_labels()
sorted_labels = sorted(vals, key=vals.get, reverse=True)
order = [labels.index(l) for l in sorted_labels]
ax.legend([handles[idx] for idx in order], [labels[idx]
                                            for idx in order], loc="upper left", bbox_to_anchor=(1, 1))
# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
# plt.show()
fig.savefig(
    f'graphs/{database}-rlsum-top1sum-{metric}.pdf', bbox_inches='tight')

# 2OP comparison
if database == "sdss":
    run_labels = ["Top1Sum_LO", "Top1Sum_LO_2OP", "Top1Sum_HI",
                  "Top1Sum_HI_2OP", "RLSum_IC", "RLSum_IC_2OP", "RLSum_DC", "RLSum_DC_2OP"]
    rlsum_variants = ["IC", "DC"]
    top1sum_variants = ["LO", "HI"]
else:
    run_labels = ["Top1Sum_IC", "Top1Sum_IC_2OP", "Top1Sum_DC",
                  "Top1Sum_DC_2OP", "RLSum_HI", "RLSum_HI_2OP", "RLSum_BL", "RLSum_BL_2OP"]
    rlsum_variants = ["HI", "BL"]
    top1sum_variants = ["IC", "DC"]
x = np.arange(len(run_labels))
utilities = []
uniformities = []
diversities = []
novelties = []

for var in top1sum_variants:
    utilities.append(runs_data["top1sum"]["all"][var]["utility"][-1])
    utilities.append(runs_data["top1sum"]["trad"][var]["utility"][-1])
    uniformities.append(runs_data["top1sum"]["all"][var]["uniformity"][-1])
    uniformities.append(runs_data["top1sum"]["trad"][var]["uniformity"][-1])
    diversities.append(runs_data["top1sum"]["all"][var]["diversity"][-1])
    diversities.append(runs_data["top1sum"]["trad"][var]["diversity"][-1])
    novelties.append(runs_data["top1sum"]["all"][var]["novelty"][-1])
    novelties.append(runs_data["top1sum"]["trad"][var]["novelty"][-1])
for var in rlsum_variants:
    utilities.append(runs_data["rlsum"]["all"][var]["utility"]["mean"][-1])
    utilities.append(runs_data["rlsum"]["trad"][var]["utility"]["mean"][-1])
    uniformities.append(runs_data["rlsum"]["all"]
                        [var]["uniformity"]["mean"][-1])
    uniformities.append(runs_data["rlsum"]["trad"]
                        [var]["uniformity"]["mean"][-1])
    diversities.append(runs_data["rlsum"]["all"][var]["diversity"]["mean"][-1])
    diversities.append(runs_data["rlsum"]["trad"]
                       [var]["diversity"]["mean"][-1])
    novelties.append(runs_data["rlsum"]["all"][var]["novelty"]["mean"][-1])
    novelties.append(runs_data["rlsum"]["trad"][var]["novelty"]["mean"][-1])

width = 0.22  # the width of the bars

fig, ax = plt.subplots(figsize=histosize)
rects1 = ax.bar(x - 1.5*width, utilities, width,
                label='utility', edgecolor='black')
rects2 = ax.bar(x - width/2, uniformities, width,
                label='uniformity', edgecolor='black', color="red", hatch='\\\\')
rects3 = ax.bar(x + width/2, diversities, width,
                label='diversity', edgecolor='black', color="green", hatch='o')
rects4 = ax.bar(x + 1.5*width, novelties,
                width, label='novelty', edgecolor='black', color="orange", hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('cumulated values')
ax.set_xticks(x)
ax.set_xticklabels(run_labels)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(
    f'graphs/{database}-2OP-utility_histogram.pdf', bbox_inches='tight')
# plt.show()


for metric in ["utility"]:  # ["uniformity", "diversity", "novelty"]:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    vals = OrderedDict()
    # for key in runs_data["rlsum"]["all"].keys():
    #     ax.plot(t, runs_data["rlsum"]["all"][key][metric]
    #             ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
    # for key in runs_data["top1sum"]["all"].keys():
    #     ax.plot(t, runs_data["top1sum"]["all"][key]
    #             [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
    if database == 'sdss':
        key = "IC"
        ax.plot(t, runs_data["rlsum"]["all"][key][metric]
                ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
        vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key][metric]["mean"][-1]
        ax.plot(t, runs_data["rlsum"]["trad"][key][metric]
                ["mean"], label=f"RLSum_{key}_2OP", marker=markers[10], markevery=5)
        vals[f"RLSum_{key}_2OP"] = runs_data["rlsum"]["trad"][key][metric]["mean"][-1]
        key = "DC"
        ax.plot(t, runs_data["rlsum"]["all"][key][metric]
                ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
        vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key][metric]["mean"][-1]
        ax.plot(t, runs_data["rlsum"]["trad"][key][metric]
                ["mean"], label=f"RLSum_{key}_2OP", marker=markers[11], markevery=5)
        vals[f"RLSum_{key}_2OP"] = runs_data["rlsum"]["trad"][key][metric]["mean"][-1]
        key = "LO"
        ax.plot(t, runs_data["top1sum"]["all"][key]
                [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
        vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key][metric][-1]
        ax.plot(t, runs_data["top1sum"]["trad"][key][metric],
                label=f"Top1Sum_{key}_2OP", marker=markers[12], markevery=5)
        vals[f"Top1Sum_{key}_2OP"] = runs_data["top1sum"]["trad"][key][metric][-1]
        key = "HI"
        ax.plot(t, runs_data["top1sum"]["all"][key]
                [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
        vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key][metric][-1]
        ax.plot(t, runs_data["top1sum"]["trad"][key][metric],
                label=f"Top1Sum_{key}_2OP", marker=markers[13], markevery=5)
        vals[f"Top1Sum_{key}_2OP"] = runs_data["top1sum"]["trad"][key][metric][-1]

    else:
        key = "HI"
        ax.plot(t, runs_data["rlsum"]["all"][key][metric]
                ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
        vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key][metric]["mean"][-1]
        ax.plot(t, runs_data["rlsum"]["trad"][key][metric]
                ["mean"], label=f"RLSum_{key}_2OP", marker=markers[10], markevery=5)
        vals[f"RLSum_{key}_2OP"] = runs_data["rlsum"]["trad"][key][metric]["mean"][-1]
        key = "BL"
        ax.plot(t, runs_data["rlsum"]["all"][key][metric]
                ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
        vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key][metric]["mean"][-1]
        ax.plot(t, runs_data["rlsum"]["trad"][key][metric]
                ["mean"], label=f"RLSum_{key}_2OP", marker=markers[11], markevery=5)
        vals[f"RLSum_{key}_2OP"] = runs_data["rlsum"]["trad"][key][metric]["mean"][-1]
        key = "IC"
        ax.plot(t, runs_data["top1sum"]["all"][key]
                [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
        vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key][metric][-1]
        ax.plot(t, runs_data["top1sum"]["trad"][key][metric],
                label=f"Top1Sum_{key}_2OP", marker=markers[12], markevery=5)
        vals[f"Top1Sum_{key}_2OP"] = runs_data["top1sum"]["trad"][key][metric][-1]
        key = "DC"
        ax.plot(t, runs_data["top1sum"]["all"][key]
                [metric], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
        vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key][metric][-1]
        ax.plot(t, runs_data["top1sum"]["trad"][key][metric],
                label=f"Top1Sum_{key}_2OP", marker=markers[13], markevery=5)
        vals[f"Top1Sum_{key}_2OP"] = runs_data["top1sum"]["trad"][key][metric][-1]

    ax.set_xlabel('pipeline length')
    ax.set_ylabel(f'cumulated {metric}')
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_labels = sorted(vals, key=vals.get, reverse=True)
    order = [labels.index(l) for l in sorted_labels]
    ax.legend([handles[idx] for idx in order], [labels[idx]
                                                for idx in order], loc="upper left", bbox_to_anchor=(1, 1))
    # ax.set_title(title)
    ax.grid(axis='y', color="gainsboro")
    fig.tight_layout()
    # plt.show()
    fig.savefig(
        f'graphs/{database}-2OP-rlsum-top1sum-{metric}.pdf', bbox_inches='tight')


# SCORE
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
vals = OrderedDict()

for key in runs_data["rlsum"]["all"].keys():
    ax.plot(t, runs_data["rlsum"]["all"][key]["score"]
            ["mean"], label=f"RLSum_{key}", marker=markers[variants.index(key)], markevery=(0+variants.index(key), 5))
    vals[f"RLSum_{key}"] = runs_data["rlsum"]["all"][key]["score"]["mean"][-1]
for key in runs_data["top1sum"]["all"].keys():
    ax.plot(t, runs_data["top1sum"]["all"][key]
            ["score"], label=f"Top1Sum_{key}", marker=markers[variants.index(key)+len(runs_data["rlsum"]["all"].keys())], markevery=(0+variants.index(key), 5))
    vals[f"Top1Sum_{key}"] = runs_data["top1sum"]["all"][key]["score"][-1]
if database == 'sdss':
    ax.plot(t, runs_data["rlsum"]["trad"]["DC"]["score"]
            ["mean"], label=f"RLSum_DC_2OP", marker=markers[10], markevery=5)
    vals[f"RLSum_DC_2OP"] = runs_data["rlsum"]["trad"]["DC"]["score"]["mean"][-1]
    ax.plot(t, runs_data["top1sum"]["trad"]["HI"]["score"],
            label=f"Top1Sum_HI_2OP", marker=markers[11], markevery=5)
    vals[f"Top1Sum_HI_2OP"] = runs_data["top1sum"]["trad"]["HI"]["score"][-1]
else:
    ax.plot(t, runs_data["rlsum"]["trad"]["BL"]["score"]
            ["mean"], label=f"RLSum_BL_2OP", marker=markers[10], markevery=5)
    vals[f"RLSum_BL_2OP"] = runs_data["rlsum"]["trad"]["BL"]["score"]["mean"][-1]
    ax.plot(t, runs_data["top1sum"]["trad"]["DC"]["score"],
            label=f"Top1Sum_DC_2OP", marker=markers[11], markevery=5)
    vals[f"Top1Sum_DC_2OP"] = runs_data["top1sum"]["trad"]["DC"]["score"][-1]
ax.set_xlabel('pipeline length')
ax.set_ylabel('quality')

# get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
sorted_labels = sorted(vals, key=vals.get, reverse=True)
order = [labels.index(l) for l in sorted_labels]
ax.legend([handles[idx] for idx in order], [labels[idx]
                                            for idx in order], loc="upper left", bbox_to_anchor=(1, 1))
# ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
# plt.show()
fig.savefig(
    f'graphs/{database}-rlsum-top1sum-quality.pdf', bbox_inches='tight')

# OPERATORS
run_labels = ["RLSum_"+x for x in variants]+["Top1Sum_"+x for x in variants]
x = np.arange(len(run_labels))
by_facets = [runs_data["rlsum"]["all"][x]["operation_counters"]["mean"]["by_facet"]
             for x in variants]+[runs_data["top1sum"]["all"][x]["operation_counters"]["by_facet"]
                                 for x in variants]
by_supersets = [runs_data["rlsum"]["all"][x]["operation_counters"]["mean"]["by_superset"]
                for x in variants]+[runs_data["top1sum"]["all"][x]["operation_counters"]["by_superset"]
                                    for x in variants]
by_neighborss = [runs_data["rlsum"]["all"][x]["operation_counters"]["mean"]["by_neighbors"]
                 for x in variants]+[runs_data["top1sum"]["all"][x]["operation_counters"]["by_neighbors"]
                                     for x in variants]
by_distributions = [runs_data["rlsum"]["all"][x]["operation_counters"]["mean"]["by_distribution"]
                    for x in variants]+[runs_data["top1sum"]["all"][x]["operation_counters"]["by_distribution"]
                                        for x in variants]
width = 0.22  # the width of the bars

fig, ax = plt.subplots(figsize=histosize)
rects1 = ax.bar(x - 1.5*width, by_facets, width,
                label='by_facet', edgecolor='black')
rects2 = ax.bar(x - width/2, by_supersets, width,
                label='by_superset', edgecolor='black', hatch='\\\\')
rects3 = ax.bar(x + width/2, by_neighborss, width,
                label='by_neighbors', edgecolor='black', hatch='o')
rects4 = ax.bar(x + 1.5*width, by_distributions,
                width, label='by_distribution', edgecolor='black', hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Uses')
ax.set_xticks(x)
ax.set_xticklabels(run_labels)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
# ax.bar_label(rects4, padding=3)

fig.tight_layout()
fig.savefig(
    f'graphs/{database}-operator_usage.pdf', bbox_inches='tight')
plt.show()

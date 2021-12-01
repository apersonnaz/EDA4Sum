import argparse
import json
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

database = 'spotify'
operators = ["by_facet", "by_superset",
                         "by_neighbors", "by_distribution"]
variants = ["IC", "LO", "BL", "HI", "DC"]
thresholds = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
markers = ["o", "v", "s", "*", "p", "P", "h", "X", "D", "+", 2, 3]
runs_data = {
    "top1sum": {

    },
}


for variant in variants:
    variant_values = []
    quality_values = []
    for threshold in thresholds:
        if threshold == "2":
            file_text = ""
        else:
            file_text = "_"+threshold
        if os.path.isfile(f"./runs-data/top1sum/{database}/all/{variant}{file_text}.json"):
            with open(f"./runs-data/top1sum/{database}/all/{variant}{file_text}.json") as f:
                data = json.load(f)
                cumulated_utility = sum([x["utility"] for x in data])
                variant_values.append(cumulated_utility)
                # quality_values.append(data[-1]["class_score_found_15"])
                quality_values.append(data[-1]["found_genre_50"])

    runs_data["top1sum"][variant] = {
        "utility": variant_values,
        "quality":
        quality_values
    }

figsize = (6, 3.5)

t = thresholds

# UTILITY
# x = np.arange(len(list(processed_data.keys())))
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()

for key in runs_data["top1sum"].keys():
    ax.plot(t, runs_data["top1sum"][key]
            ["utility"], label=f"Top1Sum_{key}", marker=markers[variants.index(key)])

ax.set_xlabel('SWAP threshold')
ax.set_ylabel('cumulated utility')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
plt.show()
fig.savefig(
    f'graphs/{database}-swap_threshold-top1sum-utility.pdf', bbox_inches='tight')

# SCORE
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()

for key in runs_data["top1sum"].keys():
    ax.plot(t, runs_data["top1sum"][key]
            ["quality"], label=f"Top1Sum_{key}", marker=markers[variants.index(key)])

ax.set_xlabel('SWAP threshold')
ax.set_ylabel('quality')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
plt.show()
fig.savefig(
    f'graphs/{database}-swap_threshold-top1sum-quality.pdf', bbox_inches='tight')

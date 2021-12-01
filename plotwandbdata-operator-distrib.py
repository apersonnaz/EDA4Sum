import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv(
    "wandb-data/all-op-25FAM-operation-distrib-evolution-training.csv")
d = d.iloc[0:2000]
prefix = "all-op-"
title = "all operator"
cols = ['FAMO', '75FAM-25CUR', '50FAM-50CUR', '25FAM-75CUR', 'CURO']
directory = f"./{prefix}graphs"
operators = ['by_facet', 'by_superset', 'by_neighbors', 'by_distribution']
t = list(range(len(d)))

data = {
    'by_facet': d['75FAM-25CUR - by_facet'].to_list(),
    'by_superset': d['75FAM-25CUR - by_superset'].to_list(),
    'by_neighbors': d['75FAM-25CUR - by_neighbors'].to_list(),
    'by_distribution': d['75FAM-25CUR - by_distribution'].to_list(),
}
# for c in cols:
#     d[f"{c}_EMA"] = d[c].ewm(span=95, adjust=True).mean()

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot()
ax.stackplot(t, data.values(),
             labels=data.keys())
# for c in cols:
#     ax.plot(t, d[c+"_EMA"].to_list(), label=c, linewidth=1)
ax.set_xlabel('episode')
ax.set_ylabel('operator occurrences')
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
# ax.set_title("all operator")
# ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig.savefig(
    f'{directory}/operator_distribution_75FAM-25CUR_all-op-training.png', bbox_inches='tight')
plt.show()

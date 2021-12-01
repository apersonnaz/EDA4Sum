import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

ds_name = 'spotify'
variants = ["IC", "LO", "BL", "HI", "DC"]
file = f"./runs-data/rlsum/{ds_name}/training_utility.csv"
markers = ["o", "v", "s", "*", "p", "P", "h", "X", "D", "+", 2, 3]
d = pd.read_csv(file)
d = d.iloc[0:4000]
t = list(range(len(d)))

for variant in variants:
    d[f"{variant}"] = d[f"{ds_name}-RLSum_{variant} - utility"].ewm(
        span=95, adjust=True).mean()
plt.rcParams["font.weight"] = "bold"
plt.rc("axes", labelweight="bold")
fig = plt.figure(figsize=(7, 2))
ax = fig.add_subplot()
for variant in variants:
    ax.plot(t, d[variant].to_list(
    ), label=f"RLSum_{variant}", linewidth=1, marker=markers[variants.index(variant)], markevery=500)
ax.set_xlabel('episodes')
ax.set_ylabel('utility')
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
# ax.set_title(title, fontweight="bold", fontsize="x-large", **hfont)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig.savefig(
    f'graphs/{ds_name}_utility_training.pdf', bbox_inches='tight')

plt.show()

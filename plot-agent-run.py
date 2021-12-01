import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

width = 0.2  # the width of the bars
title = "traditional"
with open(f"./runs-data-2op-5.json") as f:
    data = json.load(f)
prefix = '2-op'

processed_data = {}
directory = f"./{prefix}-graphs"
if not os.path.exists(directory):
    os.makedirs(directory)

operators = ['by_facet', 'by_superset', 'by_neighbors', 'by_distribution']

for model in data.keys():
    processed_data[model] = {}
    model_data = data[model]
    cumulated_rewards = []
    cumulated_curiosity_rewards = []
    operation_counters = []
    operation_reward_counters = []
    operation_cur_reward_counters = []
    viewed_sets_counters = []
    reviewed_sets_counters = []
    input_sets_size_counters = []
    for run in model_data:
        reward_sum = 0
        curiosity_reward_sum = 0
        operation_counter = [0, 0, 0, 0]
        operation_reward_counter = [0, 0, 0, 0]
        operation_cur_reward_counter = [0, 0, 0, 0]
        cumulated_reward = []
        cumulated_curiosity_reward = []
        for j in range(len(run)):
            reward_sum += run[j]["reward"]
            cumulated_reward.append(reward_sum)
            curiosity_reward_sum += run[j]["curiosity_reward"]
            cumulated_curiosity_reward.append(curiosity_reward_sum)
            operation_counter[operators.index(run[j]["operator"])] += 1
            operation_reward_counter[operators.index(
                run[j]["operator"])] += run[j]["reward"]
            operation_cur_reward_counter[operators.index(
                run[j]["operator"])] += run[j]["curiosity_reward"]
        cumulated_rewards.append(cumulated_reward)
        cumulated_curiosity_rewards.append(cumulated_curiosity_reward)
        operation_counters.append(operation_counter)
        operation_reward_counters.append(operation_reward_counter)
        operation_cur_reward_counters.append(operation_cur_reward_counter)
        viewed_sets_counters.append(list(map(lambda x: x["sets_viewed"], run)))
        reviewed_sets_counters.append(
            list(map(lambda x: x["sets_reviewed"], run)))
        input_sets_size_counters.append(
            list(map(lambda x: x["input_set_size"], run)))

    processed_data[model]["cumulated reward"] = {
        "mean": [],
        "std": []
    }
    for i in range(len(cumulated_rewards[0])):
        step_values = list(map(lambda x: x[i], cumulated_rewards))
        processed_data[model]["cumulated reward"]["mean"].append(
            np.mean(step_values))
        processed_data[model]["cumulated reward"]["std"].append(
            np.std(step_values))

    processed_data[model]["cumulated curiosity reward"] = {
        "mean": [np.mean(k) for k in zip(*cumulated_curiosity_rewards)],
        "std": [np.std(k) for k in zip(*cumulated_curiosity_rewards)]
    }

    processed_data[model]["operators count"] = {
        "mean": [np.mean(k) for k in zip(*operation_counters)],
        "std": [np.std(k) for k in zip(*operation_counters)]
    }
    processed_data[model]["operators reward sum"] = {
        "mean": [np.mean(k) for k in zip(*operation_reward_counters)],
        "std": [np.std(k) for k in zip(*operation_reward_counters)]
    }
    processed_data[model]["operators cur reward sum"] = {
        "mean": [np.mean(k) for k in zip(*operation_cur_reward_counters)],
        "std": [np.std(k) for k in zip(*operation_cur_reward_counters)]
    }

    processed_data[model]["sets viewed count"] = {
        "mean": [np.mean(k) for k in zip(*viewed_sets_counters)],
        "std": [np.std(k) for k in zip(*viewed_sets_counters)]
    }
    processed_data[model]["sets reviewed count"] = {
        "mean": [np.mean(k) for k in zip(*reviewed_sets_counters)],
        "std": [np.std(k) for k in zip(*reviewed_sets_counters)]
    }
    processed_data[model]["input set size"] = {
        "mean": [np.mean(k) for k in zip(*input_sets_size_counters)],
        "std": [np.std(k) for k in zip(*input_sets_size_counters)]
    }
    # for i in range(len(operators)):
    #     counters = list(map(lambda x: x[i], operation_counters))
    #     processed_data[model]["operators count"].append(np.mean(counters))

figsize = (6, 3.5)

t = range(len(data[list(data.keys())[0]][0]))
x = np.arange(len(list(processed_data.keys())))
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
variant_names = list(processed_data.keys())
for key in processed_data.keys():
    ax.errorbar(t, processed_data[key]["cumulated reward"]["mean"],
                yerr=processed_data[key]["cumulated reward"]["std"], linestyle='solid', label=key, errorevery=(0+variant_names.index(key), 5), elinewidth=1)
ax.set_xlabel('step')
ax.set_ylabel('cumulated familiarity reward')
ax.legend(loc="upper left")
# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig.savefig(
    f'{directory}/{prefix}-cumulated-fam-reward-online.png', bbox_inches='tight')
# fig.tight_layout()
# plt.show()

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
variant_names = list(processed_data.keys())
ax.grid(axis='y', color="gainsboro")
for key in processed_data.keys():
    ax.errorbar(t, processed_data[key]["cumulated curiosity reward"]["mean"],
                yerr=processed_data[key]["cumulated curiosity reward"]["std"], linestyle='solid', label=key, errorevery=(0+variant_names.index(key), 5), elinewidth=1)
ax.set_xlabel('step')
ax.set_ylabel('cumulated curiosity reward')
ax.legend(loc="upper left")
# ax.set_title(title)
fig.tight_layout()
fig.savefig(
    f'{directory}/{prefix}-cumulated-cur-reward-online.png', bbox_inches='tight')

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
mean_operators_counts = list(zip(
    * list(map(lambda x: processed_data[x]["operators count"]["mean"], processed_data))))
std_operators_counts = list(zip(
    * list(map(lambda x: processed_data[x]["operators count"]["std"], processed_data))))
rects1 = ax.bar(
    x - width/2, mean_operators_counts[0], width, label=operators[0], yerr=std_operators_counts[0])
rects1 = ax.bar(
    x - 1.5*width, mean_operators_counts[1], width, label=operators[1], yerr=std_operators_counts[1])
rects1 = ax.bar(
    x + width/2, mean_operators_counts[2], width, label=operators[2], yerr=std_operators_counts[2])
rects2 = ax.bar(
    x + 1.5*width, mean_operators_counts[3], width, label=operators[3], yerr=std_operators_counts[3])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('operator count')
ax.set_xticks(x)
ax.set_xticklabels(list(processed_data.keys()))
ax.legend()
# ax.set_title(title)
# ax.invert_yaxis()
# ax.xaxis.set_visible(False)
# # axs[0][1].set_xlim(0, np.sum(data, axis=1).max())
# counters_data = np.array(
#     list(map(lambda x: processed_data[x]["operators count"], processed_data)))
# category_colors = plt.get_cmap('RdYlGn')(
#     np.linspace(0.15, 0.85, counters_data.shape[1]))
# counters_data_cum = counters_data.cumsum(axis=1)
# for i, (colname, color) in enumerate(zip(operators, category_colors)):
#     widths = counters_data[:, i]
#     starts = counters_data_cum[:, i] - widths
#     ax.barh(list(processed_data.keys()), widths, left=starts, height=0.5,
#             label=colname, color=color)
#     xcenters = starts + widths / 2

#     r, g, b, _ = color
#     text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#     for y, (x, c) in enumerate(zip(xcenters, widths)):
#         ax.text(x, y, str(int(c)), ha='center', va='center',
#                 color=text_color)
# ax.legend(ncol=len(operators), bbox_to_anchor=(0, 1),
#           loc='lower left', fontsize='small', title='Operators count')
fig.tight_layout()
fig.savefig(
    f'{directory}/{prefix}-operator-distribution-online.png', bbox_inches='tight')


# fig, ax = plt.subplots(figsize=figsize)
# for key in processed_data.keys():
#     ax.errorbar(t, processed_data[key]["sets viewed count"]["mean"],
#                 yerr=processed_data[key]["sets viewed count"]["std"], linestyle='solid', label=key, errorevery=(0+variant_names.index(key), 5), elinewidth=1)
# ax.set_xlabel('step')
# ax.set_ylabel('sets viewed count')
# ax.legend()
# #ax.set_title(title)
# ax.grid(axis='y', color="gainsboro")
# fig.tight_layout()
# fig.savefig(
#     f'{directory}/{prefix}-sets-viewed-online.png', bbox_inches='tight')


# fig, ax = plt.subplots(figsize=figsize)
# mean_rewards_per_operator = list(zip(
#     * list(map(lambda x: processed_data[x]["operators reward sum"]["mean"], processed_data))))
# std_rewards_per_operator = list(zip(
#     * list(map(lambda x: processed_data[x]["operators reward sum"]["std"], processed_data))))
# rects1 = ax.bar(
#     x - width/2, mean_rewards_per_operator[0], width, label=operators[0], yerr=std_rewards_per_operator[0])
# rects1 = ax.bar(
#     x - 1.5*width, mean_rewards_per_operator[1], width, label=operators[1], yerr=std_rewards_per_operator[1])
# rects1 = ax.bar(
#     x + width/2, mean_rewards_per_operator[2], width, label=operators[2], yerr=std_rewards_per_operator[2])
# rects2 = ax.bar(
#     x + 1.5*width, mean_rewards_per_operator[3], width, label=operators[3], yerr=std_rewards_per_operator[3])
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Familiarity reward')
# ax.set_xticks(x)
# ax.set_xticklabels(list(processed_data.keys()))
# ax.legend()
# #ax.set_title(title)
# fig.tight_layout()
# fig.savefig(
#     f'{directory}/{prefix}-operator-fam-reward-distribution-online.png', bbox_inches='tight')


# fig, ax = plt.subplots(figsize=figsize)
# mean_rewards_per_operator = list(zip(
#     * list(map(lambda x: processed_data[x]["operators cur reward sum"]["mean"], processed_data))))
# std_rewards_per_operator = list(zip(
#     * list(map(lambda x: processed_data[x]["operators cur reward sum"]["std"], processed_data))))
# rects1 = ax.bar(
#     x - width/2, mean_rewards_per_operator[0], width, label=operators[0], yerr=std_rewards_per_operator[0])
# rects1 = ax.bar(
#     x - 1.5*width, mean_rewards_per_operator[1], width, label=operators[1], yerr=std_rewards_per_operator[1])
# rects1 = ax.bar(
#     x + width/2, mean_rewards_per_operator[2], width, label=operators[2], yerr=std_rewards_per_operator[2])
# rects2 = ax.bar(
#     x + 1.5*width, mean_rewards_per_operator[3], width, label=operators[3], yerr=std_rewards_per_operator[3])
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Curiosity reward')
# ax.set_xticks(x)
# ax.set_xticklabels(list(processed_data.keys()))
# ax.legend()
# #ax.set_title(title)
# fig.tight_layout()
# fig.savefig(
#     f'{directory}/{prefix}-operator-cur-reward-distribution-online.png', bbox_inches='tight')


# fig, ax = plt.subplots(figsize=figsize)
# for key in processed_data.keys():
#     ax.errorbar(t, processed_data[key]["sets reviewed count"]["mean"],
#                 yerr=processed_data[key]["sets reviewed count"]["std"], linestyle='solid', label=key, errorevery=(0+variant_names.index(key), 5), elinewidth=1)
# ax.set_xlabel('step')
# ax.set_ylabel('sets reviewed count')
# ax.legend()
# #ax.set_title(title)
# ax.grid(axis='y', color="gainsboro")
# fig.tight_layout()
# fig.savefig(
#     f'{directory}/{prefix}-sets-reviewed-online.png', bbox_inches='tight')
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot()
for key in processed_data.keys():
    ax.plot(
        t, list(map(lambda x: x["input_set_size"], data[key][0])), label=key)
ax.set_xlabel('step')
ax.set_ylabel('input set size')
ax.set_yscale("log")
ax.legend()
# ax.set_title(title)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig.savefig(
    f'{directory}/{prefix}-input-set-size-online.png', bbox_inches='tight')


plt.show()
# autolabel(rects1)
# autolabel(rects2)
# ax.invert_yaxis()
# ax.xaxis.set_visible(False)
# # axs[0][1].set_xlim(0, np.sum(data, axis=1).max())
# counters_data = np.array(
#     list(map(lambda x: processed_data[x]["operators reward sum"], processed_data)))
# category_colors = plt.get_cmap('RdYlGn')(
#     np.linspace(0.15, 0.85, counters_data.shape[1]))
# counters_data_cum = counters_data.cumsum(axis=1)
# for i, (colname, color) in enumerate(zip(operators, category_colors)):
#     widths = counters_data[:, i]
#     starts = counters_data_cum[:, i] - widths
#     ax.barh(list(processed_data.keys()), widths, left=starts, height=0.5,
#             label=colname, color=color)
#     xcenters = starts + widths / 2

#     r, g, b, _ = color
#     text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#     for y, (x, c) in enumerate(zip(xcenters, widths)):
#         ax.text(x, y, '%.2f' % c, ha='center', va='center',
#                 color=text_color)
# ax.legend(ncol=len(operators), bbox_to_anchor=(0, 1),
#           loc='lower left', fontsize='small', title='Operators reward sum')


# fig, axs = plt.subplots(3, 2)
# # axs[0][0].plot(t, list(map(lambda x: x["reward"], runs_data[0])),
# #     t, list(map(lambda x: x["reward"], runs_data[1])))
# # axs[0][0].set_xlabel('step')
# # axs[0][0].set_ylabel('reward acquired')
# # axs[0][0].grid(True)

# runs_cumulated_rewards = []

# for i in range(len(runs_data)):
#     reward_sum = 0
#     cumulated_rewards = []
#     for j in range(len(run_data)):
#         reward_sum+= runs_data[i][j]["reward"]
#         cumulated_rewards.append(reward_sum)
#     runs_cumulated_rewards.append(cumulated_rewards)

# axs[0][1].plot(t, runs_cumulated_rewards[0], t , runs_cumulated_rewards[1])
# axs[0][1].set_xlabel('step')
# axs[0][1].set_ylabel('cumulated reward')
# axs[0][1].grid(True)

# axs[1][0].plot(t, list(map(lambda x: x["input_set_size"], runs_data[0])),
#     t, list(map(lambda x: x["input_set_size"], runs_data[1])))
# axs[1][0].set_xlabel('step')
# axs[1][0].set_ylabel('input set size')
# axs[1][0].set_yscale("log")
# axs[1][0].grid(True)

# axs[1][1].plot(t, list(map(lambda x: x["output_set_count"], runs_data[0])),
#     t, list(map(lambda x: x["output_set_count"], runs_data[1])))
# axs[1][1].set_xlabel('step')
# axs[1][1].set_ylabel('output set count')
# axs[1][1].grid(True)

# axs[2][0].plot(t, list(map(lambda x: x["output_set_average_size"], runs_data[0])),
#     t, list(map(lambda x: x["output_set_average_size"], runs_data[1])))
# axs[2][0].set_xlabel('step')
# axs[2][0].set_ylabel('output set average size')
# axs[2][0].set_yscale("log")
# axs[2][0].grid(True)

# operators = ["by_facet", "by_superset", "by_neighbors", "by_distribution"]
# axs[2][1].plot(t, list(map(lambda x: x["operator"], runs_data[0])),
#     t, list(map(lambda x: x["operator"], runs_data[1])))
# axs[2][1].set_xlabel('step')
# axs[2][1].set_ylabel('Operators')
# axs[2][1].grid(True)

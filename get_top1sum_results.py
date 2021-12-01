import json
import os
import numpy as np
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

variants_data = {
    "all_data": {
        # "increasing_gamma-[0.5, 0.5, 0]": {},
        "decreasing_gamma-[0.0005, 0.0005, 0.999]": {},
        "constant-[0.45, 0.45, 0.1]": {},
        # "constant-[0.4, 0.4, 0.2]": {},
        "constant-[0.333, 0.333, 0.334]": {},
        "constant-[0.1, 0.1, 0.8]": {},
        # "greedysum": {}
    },
    # "median_size_set": {
    #     "increasing_gamma-[0.5, 0.5, 0]": {},
    #     "decreasing_gamma-[0.0005, 0.0005, 0.999]": {},
    #     "constant-[0.45, 0.45, 0.1]": {},
    #     "constant-[0.4, 0.4, 0.2]": {},
    #     "constant-[0.333, 0.333, 0.334]": {},
    #     "constant-[0.1, 0.1, 0.8]": {},
    #     "greedysum": {}
    # },
    # "smallest_set": {
    #     "increasing_gamma-[0.5, 0.5, 0]": {},
    #     "decreasing_gamma-[0.0005, 0.0005, 0.999]": {},
    #     "constant-[0.45, 0.45, 0.1]": {},
    #     "constant-[0.4, 0.4, 0.2]": {},
    #     "constant-[0.333, 0.333, 0.334]": {},
    #     "constant-[0.1, 0.1, 0.8]": {},
    #     "greedysum": {}
    # }
}

dataset = "sdss"
for filename in os.listdir(f"runs-data/{dataset}/all"):
    with open(f"runs-data/{dataset}/all/"+filename) as f:
        data = json.load(f)

    utility = [d["utility"] for d in data]
    uniformity = [d["uniformity"] for d in data]
    diversity = [d["diversity"] for d in data]
    novelty = [d["novelty"] for d in data]
    extrinsic_reward = [d["extrinsic_reward"] for d in data]
    operation_counters = Counter([d["operator"] for d in data])
    sorted_operation_counters = OrderedDict(sorted(operation_counters.items()))
    dinstinct_set_seen = data[-1]["sets_viewed"]

    distinct_states_seen = len(
        set([f'{d["operator"]}-{d["parameter"]}-{d["input_set_id"]}' for d in data]))

    print(filename)
    run_info = filename.replace('.json', '').split('-')
    start_set = run_info[0]
    variant = "-".join(run_info[1:])
    # print(f"utility: {sum(utility)}")
    # print(f"uniformity: {sum(uniformity)}")
    # print(f"diversity: {sum(diversity)}")
    # print(f"novelty: {sum(novelty)}")
    # print(f"extrinsic_reward: {sum(extrinsic_reward)}")
    # print(sorted_operation_counters)
    variants_data[start_set][variant] = {
        "utility": sum(utility),
        "utilities": utility,
        "uniformity": sum(uniformity),
        "diversity": sum(diversity),
        "novelty": sum(novelty),
        "extrinsic_reward": sum(extrinsic_reward),
        "operation_counters": operation_counters,
        "dinstinct_set_seen": dinstinct_set_seen,
        "distinct_states_seen": distinct_states_seen,
        "inverted_entropy_score": data[-1]["inverted_entropy_score"],
    }
    if dataset == "sdss":
        variants_data[start_set][variant].update({
            "class_score_found_12":  data[-1]["class_score_found_12"],
            "class_score_found_15": data[-1]["class_score_found_15"],
            "class_score_found_18": data[-1]["class_score_found_18"],
            "class_score_found_21": data[-1]["class_score_found_21"],
            "mean_class_score": data[-1]["galaxy_class_score"],
        })
    else:
        variants_data[start_set][variant].update({
            "found_genre_10": data[-1]["found_genre_10"],
            "found_genre_20": data[-1]["found_genre_20"],
            "found_genre_30": data[-1]["found_genre_30"],
            "found_genre_40": data[-1]["found_genre_40"],
            "found_genre_50": data[-1]["found_genre_50"],
            "found_genre_60": data[-1]["found_genre_60"],
            "found_genre_70": data[-1]["found_genre_70"],
            "found_genre_80": data[-1]["found_genre_80"],
            "found_genre_90": data[-1]["found_genre_90"],
            "found_genre_100": data[-1]["found_genre_100"],
        })
print(variants_data)

for start_set, runs_data in variants_data.items():

    run_labels = list(runs_data.keys())
    # run_labels.remove("greedysum")
    utilities = [runs_data[x]["utility"] for x in run_labels]
    uniformities = [runs_data[x]["uniformity"] for x in run_labels]
    diversities = [runs_data[x]["diversity"] for x in run_labels]
    novelties = [runs_data[x]["novelty"] for x in run_labels]
    extrinsic_rewards = [runs_data[x]["extrinsic_reward"] for x in run_labels]

    x = np.arange(len(run_labels))  # the label locations
    width = 0.17  # the width of the bars

    fig, ax = plt.subplots(figsize=(18, 5))
    rects1 = ax.bar(x - 2*width, utilities, width, label='utility')
    rects2 = ax.bar(x - width, uniformities, width, label='uniformity')
    rects3 = ax.bar(x, diversities, width, label='diversity')
    rects4 = ax.bar(x + width, novelties, width, label='novelty')
    rects5 = ax.bar(x + 2*width, extrinsic_rewards,
                    width, label='extrinsic_reward')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(f'Scores by top1sum variant for {start_set}')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)

    fig.tight_layout()
    fig.savefig(
        f'graphs/{start_set}-scores.png', bbox_inches='tight')
    # plt.show()

    by_facets = [runs_data[x]["operation_counters"]["by_facet"]
                 for x in run_labels]
    by_supersets = [runs_data[x]["operation_counters"]["by_superset"]
                    for x in run_labels]
    by_neighborss = [runs_data[x]["operation_counters"]["by_neighbors"]
                     for x in run_labels]
    by_distributions = [runs_data[x]["operation_counters"]
                        ["by_distribution"] for x in run_labels]
    width = 0.22  # the width of the bars

    fig, ax = plt.subplots(figsize=(18, 5))
    rects1 = ax.bar(x - 1.5*width, by_facets, width, label='by_facet')
    rects2 = ax.bar(x - width/2, by_supersets, width, label='by_superset')
    rects3 = ax.bar(x + width/2, by_neighborss, width, label='by_neighbors')
    rects4 = ax.bar(x + 1.5*width, by_distributions,
                    width, label='by_distribution')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Uses')
    ax.set_title(f'Operator usage by top1sum variant for {start_set}')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()

    fig.savefig(
        f'graphs/{start_set}-operator_usage.png', bbox_inches='tight')
    # plt.show()

    dinstinct_set_seens = [runs_data[x]["dinstinct_set_seen"]
                           for x in run_labels]
    distinct_states_seens = [runs_data[x]
                             ["distinct_states_seen"] for x in run_labels]
    fig, ax = plt.subplots(figsize=(18, 5))
    rects1 = ax.bar(x - width/2, dinstinct_set_seens,
                    width, label='dinstinct_set_seen')
    rects2 = ax.bar(x + width/2, distinct_states_seens,
                    width, label='distinct_states_seen')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('times')
    ax.set_title(f'Distinct sets and states seen for {start_set}')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    fig.savefig(
        f'graphs/{start_set}-sets_seen.png', bbox_inches='tight')
    # plt.show()
    # run_labels.append("greedysum")
    x = np.arange(len(run_labels))
    inverted_entropy_score = [runs_data[x]
                              ["inverted_entropy_score"] for x in run_labels]

    fig, ax = plt.subplots(figsize=(18, 5))
    rects1 = ax.bar(x, inverted_entropy_score,
                    width, label='inverted_entropy_score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('score')
    ax.set_title(f'Inverted entropy score for {start_set}')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.tight_layout()

    fig.savefig(
        f'graphs/{start_set}-inverted_entropy_score.png', bbox_inches='tight')

    t = np.arange(0, 50, 1)
    ticks = np.arange(0, 50, 5)
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot()
    ax.set_xticks(ticks)

    for key, data in runs_data.items():
        ax.plot(t, np.cumsum(data["utilities"]), label=key, linewidth=1)
    ax.set_xlabel('steps')
    ax.set_ylabel('utility score')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
    # ax.set_title(title, fontweight="bold", fontsize="x-large", **hfont)
    ax.grid(axis='y', color="gainsboro")
    fig.tight_layout()
    fig.savefig(
        f'graphs/{start_set}-utilities.png', bbox_inches='tight')

    if dataset == "sdss":
        mean_class_score = [runs_data[x]["mean_class_score"]
                            for x in run_labels]

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x, mean_class_score,
                        width, label='mean_class_score')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(f'mean_class_score for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)

        fig.tight_layout()

        fig.savefig(
            f'graphs/{start_set}-mean_class_score.png', bbox_inches='tight')

        class_score_found_12 = [runs_data[x]
                                ["class_score_found_12"] for x in run_labels]

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x, class_score_found_12,
                        width, label='class_score_found_12')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(f'class_score_found_12 for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)

        fig.tight_layout()

        fig.savefig(
            f'graphs/{start_set}-class_score_found_12.png', bbox_inches='tight')

        class_score_found_15 = [runs_data[x]["class_score_found_15"]
                                for x in run_labels]

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x, class_score_found_15,
                        width, label='class_score_found_15')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(f'class_score_found_15 for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)

        fig.tight_layout()

        fig.savefig(
            f'graphs/{start_set}-class_score_found_15.png', bbox_inches='tight')

        class_score_found_18 = [runs_data[x]["class_score_found_18"]
                                for x in run_labels]

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x, class_score_found_18,
                        width, label='class_score_found_18')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(f'class_score_found_18 for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)

        fig.tight_layout()

        fig.savefig(
            f'graphs/{start_set}-class_score_found_18.png', bbox_inches='tight')

        class_score_found_21 = [runs_data[x]["class_score_found_21"]
                                for x in run_labels]

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x, class_score_found_21,
                        width, label='class_score_found_21')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('score')
        ax.set_title(f'class_score_found_21 for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)

        fig.tight_layout()

        fig.savefig(
            f'graphs/{start_set}-class_score_found_21.png', bbox_inches='tight')

    else:
        found_genre_10 = [runs_data[x]["found_genre_10"] for x in run_labels]
        found_genre_20 = [runs_data[x]["found_genre_20"] for x in run_labels]
        found_genre_30 = [runs_data[x]["found_genre_30"] for x in run_labels]
        found_genre_40 = [runs_data[x]["found_genre_40"] for x in run_labels]
        found_genre_50 = [runs_data[x]["found_genre_50"] for x in run_labels]
        found_genre_60 = [runs_data[x]["found_genre_60"] for x in run_labels]
        found_genre_70 = [runs_data[x]["found_genre_70"] for x in run_labels]
        found_genre_80 = [runs_data[x]["found_genre_80"] for x in run_labels]
        found_genre_90 = [runs_data[x]["found_genre_90"] for x in run_labels]
        found_genre_100 = [runs_data[x]["found_genre_100"] for x in run_labels]

        x = np.arange(len(run_labels))  # the label locations
        width = 0.1  # the width of the bars

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x - 4.5*width, found_genre_10,
                        width, label='found_genre_10')
        rects2 = ax.bar(x - 3.5*width, found_genre_20,
                        width, label='found_genre_20')
        rects3 = ax.bar(x - 2.5*width, found_genre_30,
                        width, label='found_genre_30')
        rects4 = ax.bar(x - 1.5*width, found_genre_40,
                        width, label='found_genre_40')
        rects5 = ax.bar(x - 0.5*width, found_genre_50,
                        width, label='found_genre_50')
        rects6 = ax.bar(x + 0.5*width, found_genre_60,
                        width, label='found_genre_60')
        rects7 = ax.bar(x + 1.5 * width, found_genre_70,
                        width, label='found_genre_70')
        rects8 = ax.bar(x + 2.5*width, found_genre_80,
                        width, label='found_genre_80')
        rects9 = ax.bar(x + 3.5*width, found_genre_90,
                        width, label='found_genre_90')
        rects10 = ax.bar(x + 4.5 * width, found_genre_100,
                         width, label='found_genre_100')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores by top1sum variant for {start_set}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        ax.bar_label(rects5, padding=3)
        ax.bar_label(rects6, padding=3)
        ax.bar_label(rects7, padding=3)
        ax.bar_label(rects8, padding=3)
        ax.bar_label(rects9, padding=3)
        ax.bar_label(rects10, padding=3)

        fig.tight_layout()
        fig.savefig(
            f'graphs/{start_set}-ratio-scores.png', bbox_inches='tight')
    # galaxy_class_mean_distance = [
    #     runs_data[x]["galaxy_class_mean_distance"] for x in run_labels]

    # fig, ax = plt.subplots(figsize=(18, 5))
    # rects1 = ax.bar(x, galaxy_class_mean_distance,
    #                 width, label='galaxy_class_mean_distance')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('score')
    # ax.set_title(f'galaxy_class_mean_distance for {start_set}')
    # ax.set_xticks(x)
    # ax.set_xticklabels(run_labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)

    # fig.tight_layout()

    # fig.savefig(
    #     f'graphs/{start_set}-galaxy_class_mean_distance.png', bbox_inches='tight')

    # class_distance_found_0dot02 = [
    #     runs_data[x]["class_distance_found_0.02"] for x in run_labels]

    # fig, ax = plt.subplots(figsize=(18, 5))
    # rects1 = ax.bar(x, class_distance_found_0dot02,
    #                 width, label='class_distance_found_0.02')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('score')
    # ax.set_title(f'class_distance_found_0.02 for {start_set}')
    # ax.set_xticks(x)
    # ax.set_xticklabels(run_labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)

    # fig.tight_layout()

    # fig.savefig(
    #     f'graphs/{start_set}-class_distance_found_0.02.png', bbox_inches='tight')

    # class_distance_found_0dot04 = [
    #     runs_data[x]["class_distance_found_0.04"] for x in run_labels]

    # fig, ax = plt.subplots(figsize=(18, 5))
    # rects1 = ax.bar(x, class_distance_found_0dot04,
    #                 width, label='class_distance_found_0.04')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('score')
    # ax.set_title(f'class_distance_found_0.04 for {start_set}')
    # ax.set_xticks(x)
    # ax.set_xticklabels(run_labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)

    # fig.tight_layout()

    # fig.savefig(
    #     f'graphs/{start_set}-class_distance_found_0.04.png', bbox_inches='tight')

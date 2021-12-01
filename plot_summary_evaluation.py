import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

start_points = ["all_data"]
for start_point in start_points:
    with open(f"runs-data/spotify/all/all_data-constant-[0.333, 0.333, 0.334].json") as f:
        # with open("runs-data/all_data-constant-[0.1, 0.1, 0.8].json") as f:
        data = json.load(f)

        inverted_entropy_score = [x["inverted_entropy_score"] for x in data]
        # galaxy_class_score = [x["galaxy_class_score"] for x in data]
        # class_score_found_12 = [x["class_score_found_12"] for x in data]
        # class_score_found_15 = [x["class_score_found_15"] for x in data]
        # class_score_found_18 = [x["class_score_found_18"] for x in data]
        # class_score_found_21 = [x["class_score_found_21"] for x in data]
        mean_ratio = [x["genre_mean_ratio"] for x in data]
        found_genre_10 = [x["found_genre_10"] for x in data]
        found_genre_20 = [x["found_genre_20"] for x in data]
        found_genre_30 = [x["found_genre_30"] for x in data]
        found_genre_40 = [x["found_genre_40"] for x in data]
        found_genre_50 = [x["found_genre_50"] for x in data]
        found_genre_60 = [x["found_genre_60"] for x in data]
        found_genre_70 = [x["found_genre_70"] for x in data]
        found_genre_80 = [x["found_genre_80"] for x in data]
        found_genre_90 = [x["found_genre_90"] for x in data]
        found_genre_100 = [x["found_genre_100"] for x in data]

        t = np.arange(0, 50, 1)
        ticks = np.arange(0, 50, 5)
        fig, ax = plt.subplots(2, 1, figsize=(18, 15))
        ax[0].plot(t, inverted_entropy_score, label="inverted_entropy_score")
        ax[0].legend()
        ax[0].set_ylabel('Score')
        ax[0].set_title(f'Inverted entropy score starting from {start_point}')
        ax[0].set_xticks(ticks)
        ax[0].grid(True)
        ax[1].plot(t, mean_ratio,
                   label="mean genre ratio",)
        ax[1].plot(t, found_genre_10,
                   label="found_genre_10")
        ax[1].plot(t, found_genre_20,
                   label="found_genre_20")
        ax[1].plot(t, found_genre_30,
                   label="found_genre_30")
        ax[1].plot(t, found_genre_40,
                   label="found_genre_40")
        ax[1].plot(t, found_genre_50,
                   label="found_genre_50")
        ax[1].plot(t, found_genre_60,
                   label="found_genre_60",)
        ax[1].plot(t, found_genre_70,
                   label="found_genre_70",)
        ax[1].plot(t, found_genre_80,
                   label="found_genre_80",)
        ax[1].plot(t, found_genre_90,
                   label="found_genre_90",)
        ax[1].plot(t, found_genre_100,
                   label="found_genre_100",)
        ax[1].set_ylabel('Score or classes found')
        ax[1].set_title(
            f'Galaxy classes found on uniformity/distance score starting from {start_point}')
        # ax[1].plot(t, galaxy_class_score,
        #            label="mean score to galaxy classes",)
        # ax[1].plot(t, class_score_found_12,
        #            label="classes found with score > 12")
        # ax[1].plot(t, class_score_found_15,
        #            label="classes found with score > 15",)
        # ax[1].plot(t, class_score_found_18,
        #            label="classes found with score > 18",)
        # ax[1].plot(t, class_score_found_21,
        #            label="classes found with score > 21",)
        # ax[1].set_ylabel('Score or classes found')
        # ax[1].set_title(
        #     f'Galaxy classes found on uniformity/distance score starting from {start_point}')
        ax[1].set_xticks(ticks)
        ax[1].grid(True)
        plt.legend()
        fig.tight_layout()
        plt.show()
        print(data[-1])

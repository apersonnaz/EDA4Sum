import json
import random
from os import listdir


class TargetSetGenerator:
    @staticmethod
    def get_diverse_target_set(database_name, number_of_samples=10):
        if database_name == "sdss":
            initial_target_items = []
            for file in listdir("./rl/targets/sdss"):
                with open("./rl/targets/sdss/"+file) as f:
                    items = json.load(f)
                    if len(items) > number_of_samples:
                        initial_target_items += random.choices(
                            items, k=number_of_samples)
                    else:
                        initial_target_items += items
            return set(initial_target_items)
        else:
            initial_target_items = []
            for file in listdir("./rl/targets/spotify"):
                with open("./rl/targets/spotify/"+file) as f:
                    items = json.load(f)
                    if len(items) > number_of_samples:
                        initial_target_items += random.choices(
                            items, k=number_of_samples)
                    else:
                        initial_target_items += items
            return set(initial_target_items)

    @staticmethod
    def get_concentrated_target_set():
        target_files = listdir("./rl/targets/")
        while True:
            target_file = random.choice(target_files)
            with open("./rl/targets/"+target_file) as f:
                items = json.load(f)
            if len(items) > 1500 and len(items) < 6000:
                # print(target_file)
                break
        return set(items)

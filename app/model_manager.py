import tensorflow as tf
import numpy as np
import json
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.state_encoder import StateEncoder
from rl.A3C_2_actors.action_manager import ActionManager


class ModelManager:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets):
        self.pipeline = pipeline
        self.action_manager = ActionManager(self.pipeline, operators=[
                                            "by_facet", "by_superset", "by_neighbors", "by_distribution"])
        self.variants = ["balanced", "decreasing", "increasing", "high", "low"]
        self.models = {}
        self.lstm_steps = 3
        self.counter_curiosity_factor = 100/250

        # self.models["Scattered"] = {}
        # with open(f"./app/app_models/Scattered/0.0/set_op_counters.json") as f:
        #     set_op_counters = json.load(f)
        #     self.models["Scattered"][0.0] = {
        #         "set": tf.keras.models.load_model(f'./app/app_models/Scattered/0.0/set_actor'),
        #         "operation": tf.keras.models.load_model(f'./app/app_models/Scattered/0.0/operation_actor'),
        #         "set_op_counters": set_op_counters
        #     }

        for variant in self.variants:
            with open(f"./app/app_models/{variant}/set_op_counters.json") as f:
                set_op_counters = json.load(f)
                self.models[variant] = {
                    "set": tf.keras.models.load_model(f'./app/app_models/{variant}/set_actor'),
                    "operation": tf.keras.models.load_model(f'./app/app_models/{variant}/operation_actor'),
                    "set_op_counters": set_op_counters
                }

    def get_prediction(self, datasets, variant, target_items, found_items_with_ratio, previous_set_states=None, previous_operation_states=None):
        state_encoder = StateEncoder(
            self.pipeline, target_items=target_items, found_items_with_ratio=found_items_with_ratio)

        set_state, reward = state_encoder.encode_datasets(datasets=datasets)

        set_actor = self.models[variant]["set"]
        operation_actor = self.models[variant]["operation"]

        if previous_set_states == None:
            previous_set_states = [
                [-1]*len(state_encoder.set_description)*self.pipeline.discrete_categories_count]*self.lstm_steps

        new_set_states = previous_set_states
        new_set_states.pop(0)
        new_set_states.append(set_state)

        set_probs = set_actor.predict(np.array(new_set_states).reshape((1, self.lstm_steps, len(
            state_encoder.set_description)*self.pipeline.discrete_categories_count)))[0]
        print(set_probs)
        set_probs = self.action_manager.fix_possible_set_action_probs(
            datasets, set_probs)
        print(set_probs)
        set_action = np.random.choice(
            self.action_manager.set_action_dim, p=set_probs)
        operation_state, dummy_reward = state_encoder.encode_dataset(
            datasets[set_action], get_reward=False)

        if previous_operation_states == None:
            previous_operation_states = [
                [-1]*len(state_encoder.set_description)]*self.lstm_steps

        new_operation_states = previous_operation_states
        new_operation_states.pop(0)
        new_operation_states.append(operation_state)

        operation_probs = operation_actor.predict(np.array(
            new_operation_states).reshape((1, self.lstm_steps, len(state_encoder.set_description))))[0]
        print(operation_probs)
        operation_probs = self.action_manager.fix_possible_operation_action_probs(
            datasets[set_action], operation_probs)
        print(operation_probs)
        operation_action = np.random.choice(
            self.action_manager.operation_action_dim, p=operation_probs)
        print(self.action_manager.set_action_types[operation_action])

        action_array = self.action_manager.set_action_types[operation_action].split(
            '-&-')
        operation = action_array[0]
        if len(action_array) > 1:
            dimension = action_array[1].replace('galaxies.', '')
        else:
            dimension = None

        if datasets[set_action].set_id != None:
            set_id = int(datasets[set_action].set_id)
        else:
            set_id = None

        return {
            "predictedOperation": operation,
            "predictedDimension": dimension,
            "predictedSetId": set_id,
            "foundItemsWithRatio": state_encoder.found_items_with_ratio,
            "setStates": new_set_states,
            "operationStates": new_operation_states,
            "reward": reward
        }

    def get_curiosity_reward(self, target_set, curiosity_weight, dataset, attribute, operator):
        if dataset.set_id == None or dataset.set_id == -1:
            set_op_pair = f"-1:{str([operator, attribute])}"
        elif attribute == None:
            set_op_pair = f"{dataset.set_id}:{str([operator])}"
        else:
            set_op_pair = f"{dataset.set_id}:{str([operator, attribute])}"
        set_op_counters = self.models[target_set][curiosity_weight]["set_op_counters"]
        if set_op_pair in set_op_counters:
            set_op_counters[set_op_pair] += 1
        else:
            set_op_counters[set_op_pair] = 1

        op_counter = set_op_counters[set_op_pair]
        return self.counter_curiosity_factor/op_counter

import argparse
import os
import json
import random
import traceback
import shutil
from copy import Error
from datetime import datetime
from threading import Lock, Thread

import numpy as np
from scipy.stats.stats import mode
import tensorflow as tf
import wandb
from tqdm import tqdm
from app.greedy_summarizer import GreedySummarizer

from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

from .critic import Critic
from .intrinsic_curiosity_model import IntrinsicCuriosityForwardModel
from .operation_actor import OperationActor
from .pipeline_environment import PipelineEnvironment
from .set_actor import SetActor

tf.keras.backend.set_floatx('float64')

now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sdss")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=20)
parser.add_argument('--actor_lr', type=float, default=0.00003)
parser.add_argument('--critic_lr', type=float, default=0.00003)
parser.add_argument('--icm_lr', type=float, default=0.05)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--lstm_steps', type=int, default=3)
parser.add_argument('--target_set', type=str, default=None)
parser.add_argument('--notes', type=str, default="")
parser.add_argument('--mode', type=str, default="scattered")
parser.add_argument('--utility_mode', type=str, default=None)
parser.add_argument('--utility_weights', nargs='+',
                    type=float, default=[0.333, 0.333, 0.334])
parser.add_argument('--curiosity_ratio', type=float, default="0")
parser.add_argument('--counter_curiosity_ratio', type=float, default="0")
parser.add_argument('--name', type=str,
                    default="")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_step', type=int, default=None)
parser.add_argument('--operators', nargs='+', type=str,
                    default=["by_facet", "by_superset", "by_neighbors", "by_distribution"])
parser.add_argument('--eval_interval', type=int, default=10)
args = parser.parse_args()
print(args.utility_weights)
if args.resume_step != None:
    args.resume = True
if args.name == "":
    if args.curiosity_ratio != 0:
        args.name = f"{args.mode}-icm-{args.curiosity_ratio}-lstm-{args.lstm_steps}-alr-{args.actor_lr}-clr-{args.critic_lr}-icmlr-{args.icm_lr}-{now.strftime('%m%d%Y_%H%M%S')}"
    elif args.counter_curiosity_ratio != 0:
        args.name = f"{args.mode}-ccur-{args.counter_curiosity_ratio}-lstm-{args.lstm_steps}-alr-{args.actor_lr}-clr-{args.critic_lr}-{now.strftime('%m%d%Y_%H%M%S')}"
    else:
        if len(args.operators) == 2:
            args.name = f"{args.dataset}-trad-{args.utility_mode}-{args.utility_weights}-{args.lstm_steps}-{now.strftime('%m%d%Y_%H%M%S')}"
        else:
            args.name = f"{args.dataset}-{args.utility_mode}-{args.utility_weights}-{args.lstm_steps}-{now.strftime('%m%d%Y_%H%M%S')}"


if not args.resume:
    args.id = wandb.util.generate_id()
    if not os.path.exists("saved_models/" + args.name):
        os.makedirs("saved_models/" + args.name)
    with open("saved_models/" + args.name + "/info.json", 'w') as f:
        json.dump(vars(args), f, indent=1)
else:
    with open("./saved_models/"+args.name+"/info.json") as f:
        items = json.load(f)
        for key in items.keys():
            if key != "resume" and key != "resume_step":
                setattr(args, key, items[key])

wandb.init(name=args.name, project="deep-rl-tf2", id=args.id, resume=args.resume, config={
    "dataset": args.dataset,
    "gamma": args.gamma,
    "update_interval": args.update_interval,
    "actor_lr": args.actor_lr,
    "critic_lr": args.critic_lr,
    "icm_lr": args.icm_lr,
    "workers": args.workers,
    "lstm_steps": args.lstm_steps,
    "target_set": args.target_set,
    "notes": args.notes,
    "mode": args.mode,
    "curiosity_ratio": args.curiosity_ratio,
    "counter_curiosity_ratio": args.counter_curiosity_ratio,
    "operators": args.operators,
    "utility_mode": args.utility_mode,
    "utility_weights": args.utility_weights
})
if args.resume_step != None:
    CUR_EPISODE = args.resume_step + 1
else:
    CUR_EPISODE = 0


class Agent:
    def __init__(self, env_name):
        data_folder = "./app/data/"
        if args.dataset == 'sdss':
            self.pipeline = PipelineWithPrecalculatedSets(
                "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10,
                exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r",
                                     "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"],
                id_column="galaxies.objID")
        elif args.dataset == 'spotify_7c':
            ds_name = "tracks_7c"
            columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
                       f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
                       f"{ds_name}.liveness"]  # , f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=True, id_column=f"{ds_name}.track_id")
        elif args.dataset == 'spotify_3c':
            ds_name = "tracks_3c"
            columns = [f"{ds_name}.popularity",
                       f"{ds_name}.acousticness", f"{ds_name}.danceability", ]
            #f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
            # f"{ds_name}.liveness" , f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=True, id_column=f"{ds_name}.track_id")
        elif args.dataset == 'spotify_20b':
            ds_name = "tracks_20b"
            columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
                       f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
                       f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=20, min_set_size=20, exploration_columns=columns, scale_values=True, id_column=f"{ds_name}.track_id")
        elif args.dataset == 'spotify_5b':
            ds_name = "tracks_5b"
            columns = [f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
                       f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
                       f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", [ds_name], data_folder=data_folder, discrete_categories_count=5, min_set_size=20, exploration_columns=columns, scale_values=True, id_column=f"{ds_name}.track_id")
        else:
            columns = ["tracks.popularity", "tracks.acousticness", "tracks.danceability",
                       "tracks.duration_ms", "tracks.energy", "tracks.instrumentalness",
                       "tracks.liveness", "tracks.loudness", "tracks.speechiness", "tracks.tempo", "tracks.valence", ]
            self.pipeline: PipelineWithPrecalculatedSets = PipelineWithPrecalculatedSets(
                "spotify", ["tracks"], data_folder=data_folder, discrete_categories_count=10, min_set_size=20, exploration_columns=columns, scale_values=True,
                id_column="tracks.track_id")

        self.steps = args.lstm_steps
        if args.mode == "scattered":
            self.episode_steps = 50
        elif args.mode == "concentrated":
            self.episode_steps = 25
        self.best_evaluation_so_far = {
            "mean_reward": 0, "mean_utility": 0, "mean_score": 0}
        self.agent_name = args.name
        self.env = PipelineEnvironment(
            self.pipeline,  target_set_name=args.target_set, mode=args.mode, episode_steps=self.episode_steps, operators=args.operators, utility_mode=args.utility_mode, utility_weights=args.utility_weights)
        self.env_name = env_name

        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim

        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        if args.resume:
            if args.resume_step != None:
                model_path = f"./saved_models/{args.name}/{args.resume_step}/"
            else:
                model_path = f"./saved_models/{args.name}/current/"
            self.global_set_actor = SetActor(
                self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=model_path+"set_actor")
            self.global_operation_actor = OperationActor(
                self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, self.agent_name, model_path=model_path+"operation_actor")
            self.global_critic = Critic(
                self.set_state_dim, self.steps, args.critic_lr, self.agent_name, model_path=model_path+"critic")
            if os.path.exists(f"{model_path}/set_op_counters.json"):
                with open(f"{model_path}/set_op_counters.json") as f:
                    self.set_op_counters = json.load(f)
            else:
                self.set_op_counters = {}
        else:
            self.global_set_actor = SetActor(
                self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, self.agent_name)
            self.global_operation_actor = OperationActor(
                self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, self.agent_name)
            self.global_critic = Critic(
                self.set_state_dim, self.steps, args.critic_lr, self.agent_name)
            self.set_op_counters = {}
        self.curiosity_module = IntrinsicCuriosityForwardModel(
            self.operation_state_dim+1, self.set_state_dim, 16, args.icm_lr, self.agent_name)
        self.num_workers = args.workers  # cpu_count()
        greedy_summarizer = GreedySummarizer(self.env.pipeline)
        uniformity_threshold = 2 if self.pipeline.database_name == "sdss" else 2
        self.startup_sets = greedy_summarizer.get_summary(
            10, uniformity_threshold, 5)

    def train(self, max_episodes=100000):
        workers = []

        for i in range(self.num_workers):
            env = PipelineEnvironment(
                self.pipeline, target_set_name=args.target_set, mode=args.mode, agentId=i, episode_steps=self.episode_steps, target_items=self.env.state_encoder.target_items, operators=args.operators, utility_mode=args.utility_mode, utility_weights=args.utility_weights)

            workers.append(WorkerAgent(
                env, self.global_set_actor, self.global_operation_actor, self.global_critic, max_episodes,
                self.curiosity_module, self.set_op_counters, agentId=i, episode_steps=self.episode_steps,
                global_best_evaluation_so_far=self.best_evaluation_so_far, startup_sets=self.startup_sets))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.global_operation_actor.save_model()
        self.global_set_actor.save_model()
        self.global_critic.save_model()


class WorkerAgent(Thread):
    def __init__(self, env: PipelineEnvironment, global_set_actor: SetActor, global_operation_actor: OperationActor,
                 global_critic: Critic, max_episodes, global_curiosity_module: IntrinsicCuriosityForwardModel,
                 global_set_op_counters, agentId=-1, episode_steps=50, global_best_evaluation_so_far=None,
                 startup_sets=None):
        Thread.__init__(self)
        self.lock = Lock()
        self.other_lock = Lock()
        self.env = env
        self.agentId = agentId
        self.set_state_dim = env.set_state_dim
        self.operation_state_dim = env.operation_state_dim
        self.steps = global_set_actor.steps
        self.set_action_dim = env.set_action_space.n
        self.operation_action_dim = env.operation_action_space.n
        self.global_set_op_counters = global_set_op_counters
        self.max_episodes = max_episodes
        self.global_set_actor = global_set_actor
        self.global_operation_actor = global_operation_actor
        self.global_best_evaluation_so_far = global_best_evaluation_so_far
        self.global_critic = global_critic
        self.episode_steps = episode_steps
        self.startup_sets = startup_sets
        self.set_actor = SetActor(
            self.set_state_dim, self.set_action_dim, self.steps, self.global_set_actor.lr, self.global_set_actor.agent_name)
        self.operation_actor = OperationActor(
            self.operation_state_dim, self.operation_action_dim, self.steps, self.global_operation_actor.lr, self.global_operation_actor.agent_name)
        self.critic = Critic(self.set_state_dim, self.steps,
                             self.global_critic.lr, self.global_critic.agent_name)
        self.target_max_curiosity_reward = 100
        self.counter_curiosity_factor = self.target_max_curiosity_reward/self.episode_steps
        if args.curiosity_ratio > 0:
            self.global_curiosity_module = global_curiosity_module
            self.curiosity_module = IntrinsicCuriosityForwardModel(
                global_curiosity_module.prediction_input_state_dim, global_curiosity_module.target_input_state_dim, global_curiosity_module.output_dim)

        self.set_actor.model.set_weights(
            self.global_set_actor.model.get_weights())
        self.operation_actor.model.set_weights(
            self.global_operation_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def evaluate(self):
        episode_rewards = 0
        episode_utilities = 0
        episode_scores = 0
        eval_length = 5 if self.env.mode == 'concentrated' else 5
        for episode_counter in range(eval_length):
            set_action_steps = [[-1] * self.set_state_dim] * self.steps
            operation_action_steps = [
                [-1] * self.operation_state_dim] * self.steps
            set_state = self.env.reset(self.startup_sets)
            set_action_steps.pop(0)
            set_action_steps.append(set_state)
            episode_set_op_counters = {}
            episode_reward = 0
            episode_extrinsic_reward = 0
            episode_intrinsic_reward = 0
            episode_uniformity = 0
            episode_diversity = 0
            episode_novelty = 0
            episode_utility = 0

            done = False
            for step_counter in tqdm(range(self.episode_steps)):
                probs = self.set_actor.model.predict(
                    np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                probs = self.env.fix_possible_set_action_probs(probs[0])
                if all(np.isnan(x) for x in probs):
                    set_action = 0
                else:
                    set_action = np.random.choice(
                        self.set_action_dim, p=probs)
                operation_state = self.env.get_operation_state(set_action)
                operation_action_steps.pop(0)
                operation_action_steps.append(operation_state)
                probs = self.operation_actor.model.predict(
                    np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                probs = self.env.fix_possible_operation_action_probs(
                    set_action, probs[0])
                if np.isnan(probs[0]):
                    operation_action = self.env.get_random_operation(
                        set_action)
                else:
                    operation_action = np.random.choice(
                        self.operation_action_dim, p=probs)

                next_set_state, reward, done, set_op_pair = self.env.step(
                    set_action, operation_action)
                if set_op_pair in episode_set_op_counters:
                    episode_set_op_counters[set_op_pair] += 1
                else:
                    episode_set_op_counters[set_op_pair] = 1
                if set_op_pair in self.global_set_op_counters:
                    op_counter = episode_set_op_counters[set_op_pair] + \
                        self.global_set_op_counters[set_op_pair]
                else:
                    op_counter = episode_set_op_counters[set_op_pair]
                episode_extrinsic_reward += self.env.extrinsic_reward
                intrinsic_reward = self.counter_curiosity_factor/op_counter
                episode_intrinsic_reward += float(intrinsic_reward)
                # print(
                #     f"EVALUATION EP: {episode_counter} Agent: {self.agentId} Op counter:{op_counter} Intrisic reward: {intrinsic_reward}")
                reward = args.counter_curiosity_ratio * \
                    float(intrinsic_reward) + \
                    (1-args.counter_curiosity_ratio) * reward
                episode_reward += reward

                episode_uniformity += self.env.uniformity
                episode_diversity += self.env.diversity
                episode_novelty += self.env.novelty
                episode_utility += self.env.utility

                set_action_steps.pop(0)
                set_action_steps.append(next_set_state)
            episode_rewards += episode_reward
            episode_utilities += episode_utility
            if self.env.pipeline.database_name == "sdss":
                episode_scores += self.env.episode_info[-1]["class_score_found_21"]
            else:
                episode_scores += self.env.episode_info[-1]["found_genre_50"]
            print('EVALUATION EP{} Agent{} EpisodeReward={}'.format(
                episode_counter, self.agentId, episode_reward))
        mean_reward = episode_rewards / eval_length
        mean_utility = episode_utilities / eval_length
        mean_score = episode_scores / eval_length
        if mean_reward > self.global_best_evaluation_so_far["mean_reward"]:
            print(f"NEW BEST EVAL {mean_reward} - MODEL SAVED")
            self.global_best_evaluation_so_far["mean_reward"] = mean_reward
            self.operation_actor.save_model(
                step="best_reward")
            self.set_actor.save_model(step="best_reward")
            self.critic.save_model(step="best_reward")
            with open(f"saved_models/{args.name}/best_reward/set_op_counters.json", 'w') as f:
                json.dump(
                    self.global_set_op_counters, f, indent=1)
        if mean_utility > self.global_best_evaluation_so_far["mean_utility"]:
            print(f"NEW BEST UTIL EVAL {mean_utility} - MODEL SAVED")
            self.global_best_evaluation_so_far["mean_utility"] = mean_utility
            self.operation_actor.save_model(
                step="best_utility")
            self.set_actor.save_model(step="best_utility")
            self.critic.save_model(step="best_utility")
            with open(f"saved_models/{args.name}/best_utility/set_op_counters.json", 'w') as f:
                json.dump(
                    self.global_set_op_counters, f, indent=1)
        if mean_score > self.global_best_evaluation_so_far["mean_score"]:
            print(f"NEW BEST SCORE EVAL {mean_score} - MODEL SAVED")
            self.global_best_evaluation_so_far["mean_score"] = mean_score
            self.operation_actor.save_model(
                step="best_score")
            self.set_actor.save_model(step="best_score")
            self.critic.save_model(step="best_score")
            with open(f"saved_models/{args.name}/best_score/set_op_counters.json", 'w') as f:
                json.dump(
                    self.global_set_op_counters, f, indent=1)

    def train(self):
        operation_episodes = 0
        set_episodes = 0
        global CUR_EPISODE
        systemRandom = random.SystemRandom()
        while self.max_episodes >= CUR_EPISODE:
            set_state_batch = []
            operation_state_batch = []
            set_action_batch = []
            operation_action_batch = []
            reward_batch = []
            icm_states_batch = []
            icm_ground_truth_batch = []
            episode_set_op_counters = {}
            episode_reward = 0
            episode_loss = 0
            episode_total_op_counters = 0
            episode_extrinsic_reward = 0
            episode_intrinsic_reward = 0
            episode_uniformity = 0
            episode_diversity = 0
            episode_novelty = 0
            episode_utility = 0
            done = False
            set_action_steps = [[-1] * self.set_state_dim] * self.steps
            operation_action_steps = [
                [-1] * self.operation_state_dim] * self.steps
            set_state = self.env.reset(self.startup_sets)
            set_action_steps.pop(0)
            set_action_steps.append(set_state)
            failed = False
            try:
                for step_counter in tqdm(range(self.episode_steps)):
                    probs = self.set_actor.model.predict(
                        np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                    probs = self.env.fix_possible_set_action_probs(probs[0])
                    if all(np.isnan(x) for x in probs):
                        set_action = 0
                    else:
                        set_action = np.random.choice(
                            self.set_action_dim, p=probs)
                    operation_state = self.env.get_operation_state(set_action)
                    operation_action_steps.pop(0)
                    operation_action_steps.append(operation_state)
                    probs = self.operation_actor.model.predict(
                        np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                    probs = self.env.fix_possible_operation_action_probs(
                        set_action, probs[0])
                    if np.isnan(probs[0]):
                        operation_action = self.env.get_random_operation(
                            set_action)
                    else:
                        operation_action = np.random.choice(
                            self.operation_action_dim, p=probs)

                    next_set_state, reward, done, set_op_pair = self.env.step(
                        set_action, operation_action)

                    if set_op_pair in episode_set_op_counters:
                        episode_set_op_counters[set_op_pair] += 1
                    else:
                        episode_set_op_counters[set_op_pair] = 1
                    next_set_action_steps = set_action_steps.copy()
                    next_set_action_steps.pop(0)
                    next_set_action_steps.append(next_set_state)
                    if set_op_pair in self.global_set_op_counters:
                        op_counter = episode_set_op_counters[set_op_pair] + \
                            self.global_set_op_counters[set_op_pair]
                    else:
                        op_counter = episode_set_op_counters[set_op_pair]
                    episode_total_op_counters += op_counter
                    if args.curiosity_ratio > 0:
                        icm_state = np.concatenate(
                            ([operation_action], operation_state))

                        loss = self.curiosity_module.get_loss(np.reshape(icm_state, [1, self.operation_state_dim+1]),
                                                              np.reshape(next_set_state, [1, self.set_state_dim]))
                        if loss > 1000000:
                            intrinsic_reward = 1
                        else:
                            intrinsic_reward = loss/1000000
                        episode_intrinsic_reward += float(intrinsic_reward)

                        # print(
                        #     f"Agent: {self.agentId} Op counter:{op_counter} Loss:{loss} Intrisic reward: {intrinsic_reward}")
                        episode_extrinsic_reward += reward
                        reward = args.curiosity_ratio * \
                            float(intrinsic_reward) + \
                            (1-args.curiosity_ratio) * reward
                        icm_states_batch.append(np.reshape(
                            icm_state, [1, self.operation_state_dim+1]))
                        icm_ground_truth_batch.append(np.reshape(
                            next_set_state, [1, self.set_state_dim]))
                        episode_loss += loss
                    else:
                        episode_extrinsic_reward += self.env.extrinsic_reward
                        intrinsic_reward = self.counter_curiosity_factor/op_counter
                        episode_intrinsic_reward += float(intrinsic_reward)
                        # print(
                        #     f" EP: {CUR_EPISODE} Agent: {self.agentId} Op counter:{op_counter} Intrisic reward: {intrinsic_reward}")
                        reward = args.counter_curiosity_ratio * \
                            float(intrinsic_reward) + \
                            (1-args.counter_curiosity_ratio) * reward
                    # else:
                    #     episode_extrinsic_reward += reward
                    episode_reward += reward

                    reward = np.reshape(reward, [1, 1])
                    reward_batch.append(reward)

                    operation_action = np.reshape(
                        operation_action, [1, 1])
                    set_action = np.reshape(set_action, [1, 1])
                    set_state_batch.append(
                        np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                    set_action_batch.append(set_action)
                    operation_state_batch.append(np.array(operation_action_steps).reshape(
                        (1, self.steps, self.operation_state_dim)))
                    operation_action_batch.append(operation_action)

                    if len(set_state_batch) >= args.update_interval or done:
                        set_states = self.list_to_batch(set_state_batch)
                        set_actions = self.list_to_batch(set_action_batch)
                        operation_states = self.list_to_batch(
                            operation_state_batch)
                        operation_actions = self.list_to_batch(
                            operation_action_batch)
                        rewards = self.list_to_batch(reward_batch)

                        next_v_value = self.critic.model.predict(
                            np.array(next_set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                        td_targets = self.n_step_td_target(
                            rewards, next_v_value, done)
                        advantages = td_targets - \
                            self.critic.model.predict(set_states)

                        with self.lock:
                            try:
                                set_actor_loss = self.global_set_actor.train(
                                    set_states, set_actions, advantages)
                                operation_actor_loss = self.global_operation_actor.train(
                                    operation_states, operation_actions, advantages)
                                critic_loss = self.global_critic.train(
                                    set_states, td_targets)

                                self.set_actor.model.set_weights(
                                    self.global_set_actor.model.get_weights())
                                # self.set_actor.global_opt_weight = self.global_set_actor.opt.weights
                                self.operation_actor.model.set_weights(
                                    self.global_operation_actor.model.get_weights())
                                # self.operation_actor.global_opt_weight = self.global_operation_actor.opt.get_weights()
                                self.critic.model.set_weights(
                                    self.global_critic.model.get_weights())
                                # self.critic.global_opt_weight = self.global_critic.opt.get_weights()
                                if args.curiosity_ratio > 0:
                                    icm_states = self.list_to_batch(
                                        icm_states_batch)
                                    icm_ground_truths = self.list_to_batch(
                                        icm_ground_truth_batch)
                                    self.global_curiosity_module.train(
                                        icm_states, icm_ground_truths)
                                    self.curiosity_module.prediction_model.set_weights(
                                        self.global_curiosity_module.prediction_model.get_weights())

                                for set_op_pair in episode_set_op_counters:
                                    if set_op_pair in self.global_set_op_counters:
                                        self.global_set_op_counters[set_op_pair] += episode_set_op_counters[set_op_pair]
                                    else:
                                        self.global_set_op_counters[set_op_pair] = episode_set_op_counters[set_op_pair]

                                save_interval = 250
                                if done and CUR_EPISODE != 0 and CUR_EPISODE % save_interval == 0:
                                    ep = CUR_EPISODE

                                    self.operation_actor.save_model(
                                        step=ep)
                                    self.set_actor.save_model(step=ep)
                                    self.critic.save_model(step=ep)
                                    with open(f"saved_models/{args.name}/{ep}/set_op_counters.json", 'w') as f:
                                        json.dump(
                                            self.global_set_op_counters, f, indent=1)
                                    # if ep - (3*save_interval) > 0 and os.path.exists(f"saved_models/{args.name}/{ep-(3*save_interval)}"):
                                    #     shutil.rmtree(
                                    #         f"saved_models/{args.name}/{ep-(3*save_interval)}")
                                if done and CUR_EPISODE != 0 and CUR_EPISODE % args.eval_interval == 0:
                                    self.evaluate()

                            except Error as error:
                                print(error)
                                traceback.print_tb(error.__traceback__)
                                print('Episode failed, retrying')
                                failed = True
                                done = True
                        set_state_batch = []
                        operation_state_batch = []
                        set_action_batch = []
                        operation_action_batch = []
                        reward_batch = []
                        icm_ground_truth_batch = []
                        icm_states_batch = []

                    episode_uniformity += self.env.uniformity
                    episode_diversity += self.env.diversity
                    episode_novelty += self.env.novelty
                    episode_utility += self.env.utility
                    set_action_steps = next_set_action_steps

                if not failed:
                    print('EP{} Agent{} EpisodeReward={}'.format(
                        CUR_EPISODE, self.agentId, episode_reward))
                    log = {
                        'reward': episode_reward,
                        'sets_viewed': len(self.env.sets_viewed),
                        'sets_reviewed': self.env.set_review_counter,
                        'min_target_found_set_size_ratio': min(self.env.state_encoder.found_items_with_ratio.values()) if len(self.env.state_encoder.found_items_with_ratio) > 0 else 0,
                        'max_target_found_set_size_ratio': max(self.env.state_encoder.found_items_with_ratio.values()) if len(self.env.state_encoder.found_items_with_ratio) > 0 else 0,
                        'avg_target_found_set_size_ratio': sum(self.env.state_encoder.found_items_with_ratio.values())/len(self.env.state_encoder.found_items_with_ratio) if len(self.env.state_encoder.found_items_with_ratio) > 0 else 0,
                        'item_found_ratio': len(self.env.state_encoder.found_items_with_ratio)/len(self.env.state_encoder.target_items),
                        'extrinsic_reward': episode_extrinsic_reward,
                        'intrisic_reward': episode_intrinsic_reward,
                        'avg_op_counter': episode_total_op_counters/self.episode_steps,
                        'uniformity': episode_uniformity,
                        'diversity': episode_diversity,
                        'novelty': episode_novelty,
                        'utility': episode_utility,
                        "inverted_entropy_score": self.env.inverted_entropy_score,
                        **self.env.operation_counter
                    }
                    if self.env.pipeline.database_name == "sdss":
                        log.update({
                            "galaxy_class_score": self.env.episode_info[-1]["galaxy_class_score"],
                            "class_score_found_15": self.env.episode_info[-1]["class_score_found_15"],
                            "class_score_found_18": self.env.episode_info[-1]["class_score_found_18"],
                            "class_score_found_21": self.env.episode_info[-1]["class_score_found_21"],
                        })
                    else:
                        log.update({
                            "genre_mean_ratio": self.env.episode_info[-1]["genre_mean_ratio"],
                            "found_genre_30": self.env.episode_info[-1]["found_genre_30"],
                            "found_genre_50": self.env.episode_info[-1]["found_genre_50"],
                            "found_genre_70": self.env.episode_info[-1]["found_genre_70"],
                        })
                    if args.curiosity_ratio > 0:
                        log['avg_loss'] = episode_loss/self.episode_steps
                    wandb.log(log)

                    with self.other_lock:
                        CUR_EPISODE += 1

                        # if CUR_EPISODE % 10 == 0:
                        #     self.global_operation_actor.save_model(
                        #         step="current")
                        #     self.global_set_actor.save_model(
                        #         step="current")
                        #     self.global_critic.save_model(
                        #         step="current")

            except Error as error:
                print(error)
                traceback.print_tb(error.__traceback__)
                print('Episode failed, retrying')

    def run(self):
        self.train()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
import os
import pickle


class Critic:
    def __init__(self, state_dim, steps, lr, agent_name, model_path=None):
        self.state_dim = state_dim
        self.steps = steps
        self.agent_name = agent_name
        self.lr = lr
        self.opt = tf.keras.optimizers.Adam(lr)
        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = tf.keras.models.load_model(model_path)
        #     if os.path.exists(model_path+"/critic_optimizer.pkl"):
        #         with open(model_path+"/critic_optimizer.pkl", 'rb') as f:
        #             self.opt.set_weights(pickle.load(f))
        # self.global_opt_weight = None

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.steps, self.state_dim)),
            Dense(1024, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            LSTM(512, return_sequences=False),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.model.trainable_variables))
        return loss

    def save_model(self, name=None, step=None):
        if name == None:
            name = self.agent_name
        if step == None:
            step = "final"
        else:
            step = str(step)
        directory = "saved_models/" + name + "/" + step + "/critic/"

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(directory)
        # with open(directory + "critic_optimizer.pkl", 'wb') as f:
        #     pickle.dump(self.global_opt_weight, f)

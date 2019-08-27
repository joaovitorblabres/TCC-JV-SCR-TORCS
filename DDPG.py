"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import math
from keras.initializers import VarianceScaling
import keras.backend as K

np.random.seed(4937)
tf.set_random_seed(3)

#####################  hyper parameters  ####################

LR_A = 0.01    # learning rate for actor
LR_C = 0.1    # learning rate for critic
GAMMA = 0.99     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.001),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 32

###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.TAU = 0.001
        self.replacement = replacement
        self.t_replace_counter = 0
        K.set_session(sess)
        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(30, 3)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(30, 3)
        self.action_gradient = tf.placeholder(tf.float32,[None, 3])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = tf.keras.Input(shape=[state_size])
        h0 = tf.keras.layers.Dense(300, activation='relu')(S)
        h1 = tf.keras.layers.Dense(600, activation='relu')(h0)
        Steering = tf.keras.layers.Dense(1,activation='tanh',kernel_initializer=VarianceScaling(scale=1e-4))(h1)
        Acceleration = tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=VarianceScaling(scale=1e-4))(h1)
        Brake = tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer=VarianceScaling(scale=1e-4))(h1)
        V = tf.keras.layers.Concatenate()([Steering,Acceleration,Brake])
        model = tf.keras.Model(inputs=S, outputs=V)
        return model, model.trainable_weights, S


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.TAU = 0.001
        self.replacement = replacement
        K.set_session(sess)

        self.model, self.action, self.state = self._build_net()
        self.target_model, self.target_action, self.target_state = self._build_net()
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def _build_net(self):
        S = tf.keras.Input(shape=[30])
        A = tf.keras.Input(shape=[3],name='action2')
        w1 = tf.keras.layers.Dense(300, activation='relu')(S)
        a1 = tf.keras.layers.Dense(600, activation='linear')(A)
        h1 = tf.keras.layers.Dense(600, activation='linear')(w1)
        h2 = tf.keras.layers.add([h1,a1])
        h3 = tf.keras.layers.Dense(600, activation='relu')(h2)
        V = tf.keras.layers.Dense(3,activation='linear')(h3)
        model = tf.keras.Model(inputs=[S,A], outputs=V)
        adam = tf.keras.optimizers.Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

#####################  Memory  ####################

from collections import deque
import random
class Memory(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

state_dim = 30
action_dim = 3
action_bound = 1.0

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

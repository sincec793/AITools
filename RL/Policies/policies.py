import tensorflow as tf
import numpy as np
from Utils.pg_utils import *



def mlp(x, hidden_sizes=[32, 16, 4,], activation=tf.tanh, output_activation=tf.nn.sigmoid):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)

def lstm(x, hidden_sizes, activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        cell = tf.nn.rnn_cell.BasicLSTMCell(h)
        x, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, activation=activation)
    x = tf.flatten(x)
    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)


POLICIES={'mlp':mlp,
          'lstm':lstm}


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space, policy_fn='mlp'):
    act_dim = a.shape.as_list()[-1]
    with tf.variable_scope(policy_fn, reuse=False):
        mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        log_std = tf.Variable(name='log_std', initial_value=-0.5*np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        #a or pi = mu(s) + std(s) dot (noise)
        #Collect a sample action
        pi = mu + tf.random_normal(tf.shape(mu))*std
        logp = gaussian_likelihood(a, mu, log_std)
        logpi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logpi

def mlp_discrete_policy(x, a, hidden_sizes, activation, output_activation, action_space, policy_fn='mlp'):
    act_dim = action_space[-1]
    with tf.variable_scope(policy_fn, reuse=False):
        logits = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
        logp = tf.reduce_sum(tf.one_hot(tf.cast(a, tf.int32), depth=act_dim) * logp_all, axis=-1)
        logpi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=-1)
    return pi, logp, logpi
#Policy Gradient Utility File
import tensorflow as tf
import numpy as np


def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)

    return rtgs

#logpi = gaussian_likelihood for stochastic environments
#x = Sampled Action
#mu = output of mlp
#log_std: random normal
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, disc_fact):
    cumsum = np.zeros_like(x)
    for i in range(len(x)):
        n = 0.0
        for val in x[i:len(x)]:
            cumsum[i] += val * disc_fact**n
            n += 1
    return cumsum

#Uses GAE-Lambda advantage calculation
#https://arxiv.org/pdf/1506.02438.pdf
def compute_adv(rews, vals, gamma, lam):
    #a_t = r_t + g*v_t+1 + v_t
    adv = np.zeros_like(rews)
    vals = np.reshape(vals, (-1,))
    for i in range(len(adv)):
        adv[i] = rews[i] + gamma * (vals[i+1] if i+1 < len(adv) else rews[i]) - vals[i]
    adv = discount_cumsum(adv, gamma * lam)
    return adv



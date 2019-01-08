#Vanilla Policy Gradient Implementation
import tensorflow as tf
import numpy as np
import os
from Utils.pg_utils import *
from RL.Policies.policies import *
class VPGBuffer:
    def __init__(self, max_len, gamma=0.99, lam=0.95):
        self.obs_buff = []
        self.act_buff = []
        self.adv_buff = []
        self.rew_buff = []
        self.ret_buff = []
        self.val_buff = []
        self.logp_buff = []
        self.gamma, self.lam = gamma, lam
        self.max_len = max_len

    def store(self, obs, act, rew, val, logp):
        self.obs_buff.append(obs)
        self.act_buff.append(act)
        self.rew_buff.append(rew)
        self.val_buff.append(val)
        self.logp_buff.append(logp)

    def compute_traj(self):
        self.adv_buff = compute_adv(self.rew_buff, self.val_buff, self.gamma, self.lam)
        self.ret_buff = discount_cumsum(self.rew_buff, self.gamma)

    def get_buff(self):
        self.compute_traj()
        adv_mean = np.mean(self.adv_buff)
        adv_std = np.std(self.adv_buff)
        #Normalize advantage
        self.adv_buff = (self.adv_buff - adv_mean)/adv_std
        return (self.obs_buff, self.act_buff, self.adv_buff, self.ret_buff, self.logp_buff)

    def clear(self):
        self.obs_buff = []
        self.act_buff = []
        self.adv_buff = []
        self.rew_buff = []
        self.ret_buff = []
        self.val_buff = []
        self.logp_buff = []



class VPG:
    def __init__(self, obs_shape, act_shape, policy='mlp', pi_lr=3e-4, vf_lr=1e-3):

        self.buffer = VPGBuffer(1000)

        self.x_ph = tf.placeholder(dtype=tf.float32, shape=obs_shape, name='X_ph')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=act_shape, name='Act_ph')

        self.adv_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='adv_ph')
        self.ret_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='ret_ph')
        self.avg_ret = tf.Variable(initial_value=0)
        pi, logp, logpi = gaussian_policy(self.x_ph, self.a_ph, (32,), tf.tanh, tf.tanh, act_shape, policy)
        val = mlp(self.x_ph, output_activation=tf.nn.relu)


        self.policy_loss = -1*tf.reduce_mean(logp * self.adv_ph)
        self.val_loss = tf.reduce_mean(tf.pow(val - tf.reshape(self.ret_ph, (-1, 1)), 2))

        tf.summary.scalar('Policy Loss', self.policy_loss)
        tf.summary.scalar('Value Est Loss', self.val_loss)

        self.get_ops = [pi, val, logpi]
        self.lr_policy = tf.Variable(pi_lr)

        self.pOpt = tf.train.AdamOptimizer(learning_rate=self.lr_policy).minimize(self.policy_loss)
        self.valOpt = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.val_loss)

    def update(self, sess, val_iters,pi_iters,  sum_writer, e):
        merge = tf.summary.merge_all()

        obs, act, adv, ret, logp = self.buffer.get_buff()
        self.avg_ret = np.mean(ret)
        tf.summary.scalar('Avg Return', self.avg_ret)
        for _ in range(pi_iters):
            sess.run(self.pOpt, feed_dict={self.x_ph:obs,
                                        self.a_ph:act,
                                        self.adv_ph:adv,
                                            })

        for _ in range(val_iters):
            sess.run(self.valOpt, feed_dict={
                self.x_ph:obs,
                self.ret_ph:ret
            })
        # Show some stats
        vloss, ploss, summ = sess.run([self.val_loss, self.policy_loss, merge], feed_dict={
            self.x_ph:obs,
            self.a_ph:act,
            self.adv_ph:adv,
            self.ret_ph:ret
        })
        sum_writer.add_summary(summ, e + 1)
        print('Avg Return: {}'.format(np.mean(ret)))
        print('Avg Value: {}'.format(np.mean(self.buffer.val_buff)))
        print('Current LR: {}'.format(self.lr_policy.eval()))
        #Singe VPG is on-policy only data collected from the most recent policy is valid
        self.buffer.clear()




    def train(self, env, n_epochs, max_steps=1000, pi_iters=100, val_iters=80, save_dir='VPG_Save', restore=False):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            else:
                if restore:
                    saver.restore(sess, os.path.join(save_dir, 'model.ckpt-405'))
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)
            sum_writer = tf.summary.FileWriter('{}/'.format(save_dir), sess.graph)
            #Run current policy in environment for # steps
            obs = env.reset()
            for n in range(n_epochs):
                print('Epoch: {}/{}'.format(n+1, n_epochs))
                for i in range(max_steps):
                    env.render()

                    obs = np.reshape(obs, (1, -1))

                    a, v_t, logp_t = sess.run(self.get_ops, feed_dict={self.x_ph:obs})
                    obs, rew, done, info = env.step(2*a[0])
                    self.buffer.store(obs, a[0], rew, v_t, logp_t)

                #When we have reached maximum steps in episode, update the policy
                self.update(sess, val_iters, pi_iters, sum_writer, n)
                #
                try:
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=n + 1)
                except:
                    print('Could not save')










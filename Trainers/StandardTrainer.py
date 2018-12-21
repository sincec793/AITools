import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time
import random
import os
class StandardTrainer:

    def train(self, sess,model, train_set, test_set, batch_size, num_epochs, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        saver = tf.train.Saver()
        sum_writer = tf.summary.FileWriter('{}/StandardTrainer/'.format(save_dir), sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        count = 0
        for i in range(num_epochs):
            data, labels = train_set
            data, labels = shuffle(data, labels)
            data, labels = np.array(data), np.array(labels)
            losses = []
            print('Epoch: {}/{}\n'.format(i+1, num_epochs))
            for x in range(0, len(data), batch_size):
                merge_sum_op = tf.summary.merge_all()
                count += 1

                _, loss, merged_sum = sess.run([model.minimize_op, model.loss_fn, merge_sum_op], feed_dict={model.input_ph:data[i:i+batch_size], model.label_ph:labels[i:i+batch_size]})
                sum_writer.add_summary(merged_sum, count)
                losses.append(np.array(loss))
                print('Step {}/{} Loss - {}\r'.format(x, len(data), loss))

            #Post epoch update
            losses = np.array(losses)

            print('Epoch Ended, Avg Loss - {}'.format(np.mean(losses, axis=0)))

            try:
                saver.save(sess, os.path.join(save_dir, 'StandardTrainer/model.ckpt'), global_step=i+1)
            except:
                print('Could not save')
        print('Training Complete...')




    def sample_dataset(self, dataset, batch_size):
        data, labels = dataset
        indices = random.sample(range(len(data)), batch_size)
        return ([data[i] for i in indices], [labels[i] for i in indices])
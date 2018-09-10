#Basic GAN network to handle images
from functools import partial

import numpy as np
import tensorflow as tf
DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
class ImageGAN:
    """
    A model for Face extraction
    """

    #Input: image array of W:28 H:28
    #Output: Flattened image
    def __init__(self, img_width, img_height, optimizer=DEFAULT_OPTIMIZER, trainGen = True, trainDiscrim = True ):

        self.input_ph = tf.placeholder(tf.float32, shape=(None, img_width, img_height))
        self.label_ph = tf.placeholder(tf.float32, shape=(None, img_width, img_height))

        lbl_trans = tf.reshape(self.label_ph, shape=(-1, img_width, img_height, 1))

        tf.summary.image('Real Image', lbl_trans)

        discrim_pred_real = self.discriminator(lbl_trans)
        self.gen_img = self.generator(self.input_ph, (img_width, img_height))
        tf.summary.image('Generated Image', self.gen_img)
        discrim_pred_fake = self.discriminator(self.gen_img)

        #Try to learn to make images that the discriminator learns to output 1
        #Below basically means that we want our fake predictions to be recognized as real predictions
        #The lambda value most likely impacts the similarity to the final image
        lam1 = 0.99
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_pred_fake,
                                                                          labels = tf.ones_like(discrim_pred_fake))) + lam1 * tf.reduce_mean(tf.abs(tf.reshape(self.label_ph, shape=(-1, img_width, img_height, 1)) - self.gen_img))


        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim_pred_real,
                                                                             labels=tf.ones_like(discrim_pred_real)))

        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim_pred_fake,
                                                                             labels=tf.zeros_like(discrim_pred_fake)))

        discrim_loss = d_loss_fake + d_loss_real

        tf.summary.scalar('Generator Loss', gen_loss)
        tf.summary.scalar('Discriminator Loss', discrim_loss)
        tf.summary.scalar('Discrim Loss Real', d_loss_real)
        tf.summary.scalar('Discrim Loss Fake', d_loss_fake)


        #self.predictions = tf.reshape(d_loss_fake, (-1, img_width, img_height))
        dOpt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, epsilon=0.1).minimize(discrim_loss)
        gOpt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_loss)
        self.loss_fn = [discrim_loss, gen_loss]

        #Allow split training of the GAN
        if(trainDiscrim and trainGen):
            self.minimize_op = tf.group(gOpt, dOpt)
        elif(trainDiscrim):
            self.minimize_op = dOpt
        elif(trainGen):
            self.minimize_op = gOpt



    #Takes an image and determines if it is real or not
    def discriminator(self, X):
        with tf.variable_scope('discrim', reuse=tf.AUTO_REUSE) as scope:
            #input = tf.reshape(X, shape=(-1, 28, 28, 1))
            conv1 = tf.layers.conv2d(X, 64, 4)
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(pool1, 32, 3)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
            conv3 = tf.layers.conv2d(pool2, 16, 2)
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
            fc1 = tf.layers.dense(pool3, units=16, activation='relu')
            out = tf.layers.dense(fc1, units=1, activation='sigmoid')
            tf.summary.histogram('Discrim Output', out)
            return out

    #Takes a noise image and produces a new image
    def generator(self, z_img, img_dims):
        with tf.variable_scope('gen', reuse=tf.AUTO_REUSE) as scope:
            z_img = tf.reshape(z_img, shape=(-1, img_dims[0], img_dims[1], 1))
            tf.summary.image('Image to Transform', z_img)
            conv2 = tf.layers.conv2d_transpose(z_img, filters=64, kernel_size=(5,5), strides=(2, 2), activation='relu')
            pool2 = tf.layers.batch_normalization(conv2)
            conv3 = tf.layers.conv2d_transpose(pool2, filters=32, kernel_size=(5,5), strides=(2, 2), activation='relu')
            pool3 = tf.layers.batch_normalization(conv3)
            conv4 = tf.layers.conv2d_transpose(pool3, filters=8, kernel_size=(5,5), strides=(2,2), activation='relu')
            pool4 = tf.layers.batch_normalization(conv4)
            #conv5 = tf.layers.conv2d_transpose(pool4, filters=16, kernel_size=1, strides=(2,2), activation='relu')
            #pool5 = tf.layers.batch_normalization(conv5)
            conv6 = tf.layers.conv2d_transpose(pool4, filters=1, kernel_size=(5,5), strides=(2,2), activation='tanh')
            pool6 = tf.layers.batch_normalization(conv6)
            flat = tf.layers.flatten(pool6)
            out = tf.layers.dense(flat, units=img_dims[0]*img_dims[1], activation='relu')
            out = tf.reshape(out, shape=(-1, img_dims[0], img_dims[1], 1))
            return out
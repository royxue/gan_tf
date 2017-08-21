import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import batch_nomarlization


class GAN():
    def __init__(self):
        self.Z_dim = 100
        self.G_w1 = 1024
        self.G_w2 = 512
        self.G_w3 = 256
        self.G_w4 = 128
        self.G_w5 = 3
        self.batch_size = 50
        self.image_shape = image_shape
        return

    def generator(self):
        with tf.variable_scope('generator') as scope:
            
        return


    def discriminator(self):
        return

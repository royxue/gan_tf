import argparse
import numpy as np
import os
import tensorflow as tf

from ops import conv2d, bn, lrelu, linear, deconv2d
from utils import load_mnist, load_data, check_folder, save_images
import time


class GAN():
    def __init__(self, sess, dataset, input_size, output_size, epoch,
                 batch, z_dim, checkpoint, result, log):
        self.sess = sess
        self.model_name = 'gan'

        self.epoch = epoch
        self.batch_size = batch
        self.z_dim = z_dim
        self.checkpoint_dir = checkpoint
        self.result_dir = result
        self.log_dir = log

        self.dateset = dataset
        self.in_h = input_size
        self.in_w = input_size
        self.out_h = output_size
        self.out_h = output_size

        self.c_dim = 1
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        self.sample_num = 64
        if (dataset == 'mnist'):
            self.data_X, self.data_y = load_mnist()
        else:
            self.data_X, self.data_y = load_data(dataset)

        self.num_batches = len(self.data_X) // self.batch_size

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            # Layer 1
            g = linear(z, 1024, scope='g_fc1')
            g = bn(g, is_training=is_training, scope='g_bn1')
            g = tf.nn.relu(g)

            # Layer 2
            g = linear(g, 128 * 7 * 7, scope='g_fc2')
            g = bn(g, is_training=is_training, scope='g_bn2')
            g = tf.nn.relu(g)

            # Construct
            g = tf.reshape(g, [self.batch_size, 7, 7, 128])
            g = deconv2d(g, [self.batch_size, 14, 14, 64],
                         4, 4, 2, 2, name='g_dc3')
            g = bn(g, is_training=is_training, scope='g_bn3')

            g = deconv2d(g, [self.batch_size, 28, 28, 1],
                         4, 4, 2, 2, name='g_dc4')
            g = tf.nn.sigmoid(g)

        return g

    def discriminator(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            d = lrelu(conv2d(inputs, 64, 4, 4, 2, 2, name='d_conv1'))
            d = lrelu(bn(conv2d(d, 128, 4, 4, 2, 2, name='d_conv2'),
                         is_training=is_training, scope='d_bn2'))
            d = tf.reshape(d, [self.batch_size, -1])
            d = lrelu(bn(linear(d, 1024, scope='d_fc3'),
                         is_training=is_training, scope='d_bn3'))
            d_logit = linear(d, 1, scope='d_fc4')
            out_d = tf.nn.sigmoid(d_logit)

            return out_d, d_logit, d

    def build(self):
        image_dims = [self.in_h, self.in_w, self.c_dim]
        bs = self.batch_size

        self.inputs = tf.placeholder(
            tf.float32, [bs] + image_dims, name='input_images')
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        d, d_logits, _ = self.discriminator(
            self.inputs, is_training=True, reuse=False)

        g = self.generator(self.z, is_training=True, reuse=False)
        d_g, d_g_logits, _ = self.discriminator(
            g, is_training=True, reuse=True)

        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                    labels=tf.ones_like(d)))
        d_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_logits,
                                                    labels=tf.zeros_like(d_g)))

        self.d_loss = d_loss + d_g_loss

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_logits,
                                                    labels=tf.ones_like(d_g)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5,
                                                  beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(
            self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1,
                                          size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter-start_epoch*self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run(
                    [self.g_optim, self.g_sum, self.g_loss],
                    feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={
                                            self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(
            checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(
            checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='The name of dataset')
    parser.add_argument('--input_size', type=int, default=28,
                        help='The size of input images')
    parser.add_argument('--output_size', type=int, default=28,
                        help='The size of output images')
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Dimension of noise vector')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='Folder to save the checkpoints')
    parser.add_argument('--result', type=str, default='results',
                        help='Folder to save the generated images')
    parser.add_argument('--log', type=str, default='logs',
                        help='Folder to save training logs')

    return check_args(parser.parse_args())


def check_args(args):
    check_folder([args.checkpoint, args.result, args.log])

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    return args


def main():
    # parse arguments
    args = parse_args()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        gan = GAN(sess, epoch=args.epoch, batch_size=args.batch_size,
                  input_size=args.input_size, output_size=args.output_size,
                  z_dim=args.z_dim, checkpoint_dir=args.checkpoint,
                  result_dir=args.result, log_dir=args.log)
        # build graph
        gan.build()
        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")
        # visualize learned generator
        gan.visualize_results(args.epoch - 1)
        print(" [*] Testing finished!")


if __name__ == '__main__':
    main()

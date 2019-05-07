import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import math
from random import randint
import cv2

class bn_placeholder(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.es = epsilon
            self.mt = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.mt, updates_collections=None, epsilon=self.es, scale=True, scope=self.name)

bn_cnt = 0
def batch_normal(x):
    global bn_cnt
    bn_obj = bn_placeholder(name=("bn" + str(bn_cnt)))
    bn_cnt += 1
    return bn_obj(x)

def func_con(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def func_decon(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", ww=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if ww:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, out_s, scope=None, stddev=0.02, bias_start=0.0, ww=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        m = tf.get_variable("Matrix", [shape[1], out_s], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [out_s],
            initializer=tf.constant_initializer(bias_start))
        if ww:
            return tf.matmul(input_, m) + b, m, b
        else:
            return tf.matmul(input_, m) + b

def load_img(image_path):

    image = cv2.imread(image_path, 1)

    img = cv2.resize(image, (256,256))
    re = np.array(img)

    return re

def get_hint(img, is_testing=False):
    if is_testing:
        img = img * 0.3 + np.ones_like(img) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0,205)
            randy = randint(0,205)
            img[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(img,(100,100))

def img_merg(images, size):
    n1, n2 = images.shape[1], images.shape[2]
    img = np.zeros((n1 * size[0], n2 * size[1], 3))

    for index, image in enumerate(images):
        i = index % size[1]
        j = index / size[1]
        img[j*n1:j*n1+n1, i*n2:i*n2+n2, :] = image

    return img

def merge(images, size):
    n1, n2 = images.shape[1], images.shape[2]
    img = np.zeros((n1 * size[0], n2 * size[1], 3))

    for index, image in enumerate(images):
        i = index % size[1]
        j = index / size[1]
        img[j*n1:j*n1+n1, i*n2:i*n2+n2, :] = image

    return img[:,:,0]

def get_hint(img, is_testing=False):
    if is_testing:
        img = img * 0.3 + np.ones_like(img) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0,205)
            randy = randint(0,205)
            img[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(img,(100,100))

def lrelu(x, l=0.2, name="lrelu"):
    return tf.maximum(x, l*x)

class Color():
    def __init__(self, imgsize=256, batchsize=1):
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = imgsize
        self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.input_colors2 = 3
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = bn_placeholder(name='d_bn1')
        self.d_bn2 = bn_placeholder(name='d_bn2')
        self.d_bn3 = bn_placeholder(name='d_bn3')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        combined_preimage = tf.concat(axis=3, values=[self.line_images, self.color_images])
        # combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        # combined_preimage = self.line_images

        self.fake_img = self.G(combined_preimage)

        self.real_AB = tf.concat(axis=3, values=[combined_preimage, self.real_images])
        self.fake_AB = tf.concat(axis=3, values=[combined_preimage, self.fake_img])

        # self.real_AB = tf.concat([combined_preimage, self.real_images], 3)
        # self.fake_AB = tf.concat([combined_preimage, self.fake_img], 3)

        self.disc_true, disc_true_logits = self.D(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.D(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        g_loss_o = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                        + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.fake_img))
        # g_loss_gan = tf.reduce_mean(-tf.log(self.d_loss_fake + 1e-12)) 
        # g_loss_l1 = tf.reduce_mean(tf.abs(self.real_images - self.fake_img))
        loss_y = tf.nn.l2_loss(self.fake_img[:, 1:, :, :] - self.fake_img[:, :-1, :, :])
        loss_x = tf.nn.l2_loss(self.fake_img[:, :, 1:, :] - self.fake_img[:, :, :-1, :])
        loss_t = 2 * (loss_y + loss_x)
        loss_t = tf.cast(loss_t, tf.float32)
        g_loss_tv = tf.reduce_mean(tf.abs(loss_t))

        mean, variance = tf.nn.moments(self.fake_img[:,:,:,:], axes=[1,2])
        print "variance: ", variance
        g_loss_v = tf.reduce_mean(variance)

        self.g_loss = 0.9*g_loss_o +  0.3*g_loss_tv /255/255 - 0.2 * g_loss_v/255/255

        t_vars = tf.trainable_variables()
        #print t_vars
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        print "++++++++++++++++++++++"
        #print "mark:::  ", self.g_vars
        print "++++++++++++++++++++++"

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

    def D(self, img, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(func_con(img, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(func_con(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(func_con(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(func_con(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def G(self, img_in):
        with tf.variable_scope("generator") as scope:
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            e1 = func_con(img_in, self.gf_dim, name='g_e1_conv')
            e2 = batch_normal(func_con(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = batch_normal(func_con(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = batch_normal(func_con(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = batch_normal(func_con(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))


            self.d4, self.d4_w, self.d4_b = func_decon(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', ww=True)
            d4 = batch_normal(self.d4)
            d4 = tf.concat(axis=3, values=[d4, e4])
            self.d5, self.d5_w, self.d5_b = func_decon(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', ww=True)
            d5 = batch_normal(self.d5)
            d5 = tf.concat(axis=3, values=[d5, e3])
            self.d6, self.d6_w, self.d6_b = func_decon(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', ww=True)
            d6 = batch_normal(self.d6)
            d6 = tf.concat(axis=3, values=[d6, e2])
            self.d7, self.d7_w, self.d7_b = func_decon(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', ww=True)
            d7 = batch_normal(self.d7)
            d7 = tf.concat(axis=3, values=[d7, e1])
            self.d8, self.d8_w, self.d8_b = func_decon(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', ww=True)

        return tf.nn.tanh(self.d8)

    def test_coloring(self,count):
        src = glob(os.path.join("imgs-sample", "*.jpg"))

        for i in range(min(100,len(src) / self.batch_size)):
            bf = src[i*self.batch_size:(i+1)*self.batch_size]
            bc = np.array([load_img(batch_file) for batch_file in bf])
            bc_ = bc/255.0

            bc_sketch = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in bc]) / 255.0
            bc_sketch = np.expand_dims(bc_sketch, 3)

            bc_hint = np.array([get_hint(ba,True) for ba in bc]) / 255.0

            r= self.sess.run(self.fake_img, feed_dict={self.real_images: bc_, self.line_images: bc_sketch, self.color_images: bc_hint})
            cv2.imwrite("sample/sample_"+str(count*100)+str(i)+".jpg",img_merg(r, [self.batch_size_sqrt, self.batch_size_sqrt]) * 255)


    def load(self, dir):
        print(" loading checkpoint")

        dir = os.path.join(dir, "tr")

        checkpoint = tf.train.get_checkpoint_state(dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.save.restore(self.sess, os.path.join(dir, checkpoint_name))
            return True
        else:
            return False

    def train(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.save = tf.train.Saver()

        if self.load("./checkpoint"):
            print "succeed"
        else:
            print "error"
        src = glob(os.path.join("imgs", "*.jpg"))
        print src[0]
        first = np.array([load_img(sample_file) for sample_file in src[0+15:self.batch_size+15]])
        first_ = first/255.0

        first_sketch = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in first]) / 255.0
        first_sketch = np.expand_dims(first_sketch, 3)

        first_hint = np.array([get_hint(ba) for ba in first]) / 255.0

        cv2.imwrite("results/first.png",255 * img_merg(first_, [self.batch_size_sqrt, self.batch_size_sqrt]))
        cv2.imwrite("results/first_sketch.jpg",255 * merge(first_sketch, [self.batch_size_sqrt, self.batch_size_sqrt]))
        cv2.imwrite("results/first_hint.jpg",255 * img_merg(first_hint, [self.batch_size_sqrt, self.batch_size_sqrt]))

        for e in xrange(1000):
            for i in range(len(src)/ self.batch_size):
                bf = src[i*self.batch_size:(i+1)*self.batch_size]
                bc = np.array([load_img(batch_file) for batch_file in bf])
                bc_ = bc/255.0

                bc_sketch = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in bc]) / 255.0
                bc_sketch = np.expand_dims(bc_sketch, 3)

                bc_hint = np.array([get_hint(ba) for ba in bc]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_optimizer], feed_dict={self.real_images: bc_, self.line_images: bc_sketch, self.color_images: bc_hint})
                g_loss, _ = self.sess.run([self.g_loss, self.g_optimizer], feed_dict={self.real_images: bc_, self.line_images: bc_sketch, self.color_images: bc_hint})

                if i % 100 == 0:
                    r= self.sess.run(self.fake_img, feed_dict={self.real_images: first_, self.line_images: first_sketch, self.color_images: first_hint})
                    cv2.imwrite("results/"+str(e*100000 + i)+".jpg",255 * img_merg(r, [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 500 == 0:
                    dir = os.path.join("./checkpoint", "tr")

                    if not os.path.exists("./checkpoint"):
                        os.makedirs("./checkpoint")

                    self.save.save(self.sess, os.path.join(dir, "model"), global_step=e*100000 + i)
                    
                print e, i, (len(src)/self.batch_size), d_loss, g_loss
            self.test_coloring(e)

    def sample(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.save = tf.train.Saver()

        if self.load("./checkpoint"):
            print "succeed"
        else:
            print "error"

        src = glob(os.path.join("imgs-valid", "*.jpg"))

        for i in range(min(100,len(src) / self.batch_size)):
            bf = src[i*self.batch_size:(i+1)*self.batch_size]
            bc = np.array([load_img(batch_file) for batch_file in bf])
            bc_ = bc/255.0

            bc_sketch = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in bc]) / 255.0
            bc_sketch = np.expand_dims(bc_sketch, 3)

            bc_hint = np.array([get_hint(ba,True) for ba in bc]) / 255.0

            r= self.sess.run(self.fake_img, feed_dict={self.real_images: bc_, self.line_images: bc_sketch, self.color_images: bc_hint})
            cv2.imwrite("test/sample_"+str(i)+".jpg",255 * img_merg(r, [self.batch_size_sqrt, self.batch_size_sqrt]))
            cv2.imwrite("test/sample_"+str(i)+"_origin.jpg",255 * img_merg(bc_, [self.batch_size_sqrt, self.batch_size_sqrt]))
            cv2.imwrite("test/sample_"+str(i)+"_line.jpg",255 * img_merg(bc_sketch, [self.batch_size_sqrt, self.batch_size_sqrt]))
            cv2.imwrite("test/sample_"+str(i)+"_color.jpg",255 * img_merg(bc_hint, [self.batch_size_sqrt, self.batch_size_sqrt]))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == "train":
        c = Color()
        c.train()
    elif cmd == "sample":
        c = Color()
        c.sample()
    else:
        print "input error"

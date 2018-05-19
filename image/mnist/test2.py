from matplotlib import pyplot as plt
from random import randint
import random
# from mnist import MNIST
import get_mnist
import cv2
import numpy as np
import tensorflow as tf 
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100 
train_images, train_labels, test_images, test_labels = get_mnist.mnist('data/')


FLAGS = None
def data_type():
#   """Return the type of the activations, weights, and placeholder variables."""
#   if FLAGS.use_fp16:
    # return tf.float16
#   else:
    return tf.float32
### init placeholders
data = tf.placeholder(
      data_type(),
      shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
### weights
conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type()))
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],stddev=0.1,seed=SEED,dtype=data_type()))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],stddev=0.1,seed=SEED,dtype=data_type()))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))
### build model
conv = tf.nn.conv2d(data,
                    conv1_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
# Bias and rectified linear non-linearity.
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
# Max pooling. The kernel size spec {ksize} also follows the layout of
# the data. Here we have a pooling window of 2, and a stride of 2.
pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
conv = tf.nn.conv2d(pool,
                    conv2_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
# Reshape the feature map cuboid into a 2D matrix to feed it to the
# fully connected layers.
pool_shape = pool.get_shape().as_list()
reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
# Fully connected layer. Note that the '+' operation automatically
# broadcasts the biases.
hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
logits = tf.matmul(hidden, fc2_weights) + fc2_biases

# sess = tf.InteractiveSession()
saver = tf.train.Saver()

path = '/home/cuong/VNG/ViettelCardReader_V5/images/'
imgs = test_images
labels = test_labels
imgs_str = 'test_images'
output_dir = imgs_str + '_output'
true_dir = './'+ output_dir + '/true_dir'
false_dir = './'+ output_dir + '/false_dir'
import os
def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
maybe_mkdir(output_dir)
maybe_mkdir(true_dir)
maybe_mkdir(false_dir)




with tf.Session() as sess:
    saver.restore(sess, 'my-model')
    for i in range(len(imgs)):
        img = imgs[i].reshape(28,28)
        label_number = np.argmax(labels[i])
        # print(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
        # sess.run(tf.global_variables_initializer())
        
        img_reshape = np.reshape(img, (1, 28, 28, 1))
        pred = sess.run(logits, feed_dict = {data: img_reshape})
        pred_number = np.argmax(pred)
        (result, write_dir) = (True, true_dir) if label_number == pred_number else (False, false_dir)
        # print(pred_number)
        # print("i = ", i)

        name_img = 'image_' + str(i) + 'is_' + str(pred_number) +'.jpg'
        # changed_img = 
        file_name = str(i) + '_is_' + str(pred_number)
        if label_number != pred_number:
            file_name += '>label_' + str(label_number)
        file_name += '_' +str(result) +'.png'
        cv2.imwrite(os.path.join(write_dir, file_name), 255*img)



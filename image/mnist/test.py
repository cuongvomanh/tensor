from matplotlib import pyplot as plt
from random import randint
import random
# from mnist import MNIST
import get_mnist
import cv2
import numpy as np
train_images, train_labels, test_images, test_labels = get_mnist.mnist('data/')
# mndata = MNIST('data')
# images, labels = mndata.load_training()
# print(images)
# num = randint(0, mndata.test.images.shape[0])
# img = mndata.test.images[num]
# img = images[0]
# index = random.randrange(0, len(test_images))  # choose an index ;-)
# print(mndata.display(test_images[index]))
img = test_images[0].reshape(28,28)
print(img.shape)
# img =np.array(img)

# cv2.imshow('img', img.reshape(28,28))
# cv2.waitKey(0)
import tensorflow as tf 
sess = tf.Session()
saver = tf.train.import_meta_graph('my-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
# saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./my-model.meta")
# num = randint(0, mnist.test.images.shape[0])
# img = mnist.test.images[num]

classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
print('NN predicted', classification[0])
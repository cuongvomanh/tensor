from matplotlib import pyplot as plt
from random import randint
import mnist
import tensorflow as tf 
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./my-model")
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]

classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
print('NN predicted', classification[0])
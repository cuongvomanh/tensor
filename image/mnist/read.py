from mnist import MNIST
import random
mndata = MNIST('./data')

images, labels = mndata.load_training()
# or
# images, labels = mndata.load_testing()
index = random.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
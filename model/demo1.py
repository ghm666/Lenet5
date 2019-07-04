from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
print(x_train.shape)

# x_train = np.expand_dims(x_train, axis=1)
# print(x_train.shape)
# mnist
MNIST in Keras

This is a re-implementation of the MNIST example that comes with TensorFlow from straight TensorFlow to Keras.

The TensorFlow code I re-implemented is:

models/tutorials/image/mnist/convolutional.py

The model implemented is exactly the same: A 2D convolutional layer followed by a max pooling layer, followed by another 2D convolutional layer followed by another max pooling layer, followed by two fully connected layers with dropout in between them.


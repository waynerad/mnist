"""This is a re-implementation of the MNIST example that comes with TensorFlow from straight
TensorFlow to Keras.

The TensorFlow code I re-implemented is:

models/tutorials/image/mnist/convolutional.py

The model implemented is exactly the same: A 2D convolutional layer followed by a max
pooling layer, followed by another 2D convolutional layer followed by another max pooling
layer, followed by two fully connected layers with dropout in between them.

The function create_keras_model is where all the magic happens.
"""

# import keras
# import tensorflow as tf
# import os
# import numpy as np
# import gzip

def maybe_download(source_url, work_directory, filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(source_url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images, image_size, num_channels, pixel_depth):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images * num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (pixel_depth / 2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def create_keras_model(image_height, image_width, num_channels, num_labels, train, randseed):
    """Here's where our modifications begin. We're just going to do a "Sequential" model
    because our model here is super simple. As you'll see, it's very simple in Keras to
    create the layers: we just do 1 line per layer: Conv2D, MaxPooling2D, Conv2D,
    MaxPooling2D, and then Dense for the fully connected last layer, with a Flatten in
    between to go from 2D to 1D for the fully connected last layer."""
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same',
                                  activation='relu', input_shape=(image_height, image_width,
                                                                  num_channels)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    if train:
        model.add(keras.layers.Dropout(0.5, seed=randseed))
    model.add(keras.layers.Dense(num_labels, activation='softmax'))
    return model

def fake_data(num_images, image_size, num_channels):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = np.ndarray(
        shape=(num_images, image_size, image_size, num_channels),
        dtype=np.float32)
    labels = np.zeros(shape=(num_images,), dtype=np.int64)
    for image in range(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels

def main():
    """Main starting point. We pull the data from Yann LeCun's website and break it into
    training, test, and validation sets, and train our model."""
    src = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    wrk = 'data'

    # Get the data.
    train_data_filename = maybe_download(src, wrk, 'train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download(src, wrk, 'train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download(src, wrk, 't10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download(src, wrk, 't10k-labels-idx1-ubyte.gz')
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000, 28, 1, 255)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000, 28, 1, 255)
    test_labels = extract_labels(test_labels_filename, 10000)
    # Generate a validation set.
    validation_size = 5000
    validation_data = train_data[:validation_size, ...]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, ...]
    train_as_nums = train_labels[validation_size:]
    train_labels = keras.utils.np_utils.to_categorical(train_as_nums, 10)
    num_epochs = 10
    eval_batch_size = 64
    train_size = train_labels.shape[0]

    train_model = create_keras_model(28, 28, 1, 10, True, 66478)
    train_model.summary()
    momentum = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    train_model.compile(loss='categorical_crossentropy', optimizer=momentum)

    train_model.fit(x=train_data, y=train_labels, batch_size=eval_batch_size, epochs=1,
                    verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
                    shuffle=True)


main()

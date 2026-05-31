import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from random import randrange



def fancy_preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0

    x = tf.pad(x, [[4,4],[4,4],[0,0]])
    x = tf.image.random_crop(x, [32,32,3])

    x = tf.image.random_flip_left_right(x)

    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)

    return x, y


def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    seed = randrange(1, 1000000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_train = x_train[..., None]
    x_train = np.repeat(x_train, 3, axis=-1)
    x_train = tf.image.resize(x_train, (32, 32)).numpy()
    x_test = x_test[..., None]
    x_test = np.repeat(x_test, 3, axis=-1)
    x_test = tf.image.resize(x_test, (32, 32)).numpy()

    tf_x_train = x_train.astype(np.float32)
    tf_x_test = x_test.astype(np.float32)

    tf_train_ds = tf.data.Dataset.from_tensor_slices((tf_x_train, y_train))
    tf_train_ds = tf_train_ds.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    tf_train_ds = tf_train_ds.cache().prefetch(tf.data.AUTOTUNE)

    return tf_train_ds, x_test, y_test

import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_normalized_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    mean = x_train.mean(axis=(0, 1, 2))
    std = x_train.std(axis=(0, 1, 2))

    def normalize(x, y):
        x = (x - mean) / std
        return tf.cast(x, tf.float32), tf.cast(y, tf.int32)

    x_train, y_train = normalize(x_train, y_train)
    x_test, y_test = normalize(x_test, y_test)

    tf_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .map(_transform_train_as_in_a_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf_test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tf_train_ds, tf_test_ds, mean, std


def load_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    tf_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(lambda x, y: (keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)))
        .cache()
        .shuffle(50000)
        .map(_transform_train_as_in_a_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    tf_test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tf_train_ds, tf_test_ds


def load_cifar_awp_split_dataset():
    subset_size = 25000
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    tf_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[:subset_size], y_train[:subset_size]))
        .map(lambda x, y: (keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)))
        .cache()
        .shuffle(25000)
        .map(_transform_train_as_in_a_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf_awp_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[subset_size:], y_train[subset_size:]))
        .map(lambda x, y: (keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)))
        .cache()
        .shuffle(25000)
        .map(_transform_train_as_in_a_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf_test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tf_train_ds, tf_awp_train_ds, tf_test_ds


def load_cifar_labels():
    return {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }


def _transform_train_as_in_a_research_paper(image, label):
    image = tf.pad(
        image,
        paddings=[[4, 4], [4, 4], [0, 0]],
        mode="CONSTANT",
        constant_values=0
    )
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label

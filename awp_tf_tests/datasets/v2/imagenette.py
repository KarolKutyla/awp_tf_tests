import tensorflow as tf
import tensorflow_datasets as tfds



def load_imagenette_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        "imagenette/320px",
        split=["train", "validation"],
        as_supervised=True,
        with_info=True
    )
    train_ds = (
        ds_train
        .shuffle(10_000)
        .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        ds_test
        .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


def preprocess_train(image, label):
    IMG_SIZE = 224

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image / 127.5 - 1.0

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, label


def preprocess_test(image, label):
    IMG_SIZE = 224

    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 127.5 - 1.0
    return image, label

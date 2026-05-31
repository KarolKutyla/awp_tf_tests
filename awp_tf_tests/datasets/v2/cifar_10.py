import tensorflow as tf



def load_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    tf_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .map(_transform_train_as_in_a_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    tf_test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tf_train_ds, tf_test_ds


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
    image = tf.cast(image, tf.float32)
    image = tf.pad(
        image,
        paddings=[[4, 4], [4, 4], [0, 0]],
        mode="CONSTANT",
        constant_values=0
    )
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, tf.cast(label, tf.int32)

import tensorflow as tf
from tensorflow.keras import layers


def preact_resnet_block(x, filters, stride=1):
    bn1 = layers.BatchNormalization()(x)
    act1 = layers.Activation("relu")(bn1)

    shortcut = x

    y = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(act1)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    y = layers.Conv2D(filters, 3, 1, padding="same", use_bias=False)(y)

    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, stride, use_bias=False)(act1)

    out = layers.Add()([y, shortcut])
    return out


def make_stage(x, filters, blocks, stride):
    x = preact_resnet_block(x, filters, stride=stride)
    for _ in range(1, blocks):
        x = preact_resnet_block(x, filters, stride=1)
    return x


def PreActResNet18(input_shape=(32, 32, 3), num_classes=10, width_mult=10):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)

    x = make_stage(x, 64 * width_mult, 2, stride=1)
    x = make_stage(x, 128 * width_mult, 2, stride=2)
    x = make_stage(x, 256 * width_mult, 2, stride=2)
    x = make_stage(x, 512 * width_mult, 2, stride=2)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs, outputs, name=f"PreActResNet18_w{width_mult}")
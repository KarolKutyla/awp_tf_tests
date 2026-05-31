import tensorflow as tf
import keras_cv
from tensorflow.keras import layers

from awp_tf_tests.models.implementations import preact_resnet_18



def load_preact_resnet_18(steps_per_epoch):
    model = preact_resnet_18.PreActResNet18(
        input_shape=(32, 32, 3),
        num_classes=10,
        width_mult=10
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    model.name = "resnet_18_custom_implementation"
    return model


def load_tensorflow_resnet_18_v1(steps_per_epoch):
    backbone = keras_cv.models.ResNet18Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    model.name = "resnet_18_v1"

    return model


def load_tensorflow_resnet_50_v2(steps_per_epoch):
    backbone = keras_cv.models.ResNet50V2Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )
    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    model.name = "resnet_50v2"

    return model


def load_tensorflow_resnet_101(steps_per_epoch):
    backbone = keras_cv.models.ResNet101V2Backbone(
        include_rescaling=False,
        input_shape=(224, 224, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    model.name = "resnet_101v2"

    return model


def load_tensorflow_resnet_152_v2(steps_per_epoch):
    backbone = keras_cv.models.ResNet152V2Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )
    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)

    model.name = "resnet_152v2"

    return model

import tensorflow as tf
import keras_cv


def load_tensorflow_resnet_18_v2_for_normal_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_normal"
    return model


def load_tensorflow_resnet_18_v2_for_adversarial_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_adversarial"
    return model


def load_tensorflow_resnet_18_v2_for_awp_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_awp"
    return model


def load_tensorflow_resnet_18_v2_for_awp_training_with_alternate_iterations(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_awp_alternate"
    return model


def _load_tensorflow_resnet_18_v2(steps_per_epoch):
    backbone = keras_cv.models.ResNet18V2Backbone(include_rescaling=False, input_shape=(32, 32, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001],
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    return model

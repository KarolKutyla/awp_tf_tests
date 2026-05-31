import tensorflow as tf

from awp_tf_tests.models.implementations import wide_resnet_28



def load_wide_resnet_standard(steps_per_epoch):
    classifier = _load_wide_resnet(steps_per_epoch)
    classifier.name = "wide_resnet_standard_training"
    return classifier


def load_wide_resnet_adversarial(steps_per_epoch):
    classifier = _load_wide_resnet(steps_per_epoch)
    classifier.name = "wide_resnet_adversarial_training"
    return classifier


def load_wide_resnet_awp(steps_per_epoch):
    classifier = _load_wide_resnet(steps_per_epoch)
    classifier.name = "wide_resnet_awp"
    return classifier


def load_wide_resnet_awp_alternate_step(steps_per_epoch):
    classifier = _load_wide_resnet(steps_per_epoch)
    classifier.name = "wide_resnet_awp_alternate"
    return classifier


def _load_wide_resnet(steps_per_epoch):
    model = wide_resnet_28.wideresnet((32, 32, 3), 10)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False, weight_decay=5e-4)
    model.compile(loss=loss, optimizer=optimizer)
    return model

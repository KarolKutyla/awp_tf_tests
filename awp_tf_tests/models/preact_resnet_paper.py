import tensorflow as tf


class PreActBlock(tf.keras.layers.Layer):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
        )

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = tf.keras.layers.Conv2D(
                filters=self.expansion * planes,
                kernel_size=1,
                strides=stride,
                padding="valid",
                use_bias=False,
            )
        else:
            self.shortcut = None

    def call(self, x, training=None):
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)

        # IMPORTANT:
        # PreAct ResNet applies the shortcut to the activated input.
        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)

        out = out + shortcut

        return out


class PreActBottleneck(tf.keras.layers.Layer):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=planes,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
        )

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
        )

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.expansion * planes,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = tf.keras.layers.Conv2D(
                filters=self.expansion * planes,
                kernel_size=1,
                strides=stride,
                padding="valid",
                use_bias=False,
            )
        else:
            self.shortcut = None

    def call(self, x, training=None):
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)

        out = self.bn3(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv3(out)

        out = out + shortcut

        return out


class PreActResNet(tf.keras.Model):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        self.in_planes = 64

        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )

        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
        )

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
        )

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
        )

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=4,
        )

        self.flatten = tf.keras.layers.Flatten()

        self.linear = tf.keras.layers.Dense(
            num_classes,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                )
            )

            self.in_planes = planes * block.expansion

        return layers

    def call(self, x, training=None):
        out = self.conv1(x)

        for block in self.layer1:
            out = block(out, training=training)

        for block in self.layer2:
            out = block(out, training=training)

        for block in self.layer3:
            out = block(out, training=training)

        for block in self.layer4:
            out = block(out, training=training)

        out = self.bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)

        out = self.flatten(out)

        out = self.linear(out)

        return out


def PreActResNet18(num_classes=10):
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
    )


def PreActResNet34(num_classes=10):
    return PreActResNet(
        PreActBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
    )


def PreActResNet50(num_classes=10):
    return PreActResNet(
        PreActBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
    )


def PreActResNet101(num_classes=10):
    return PreActResNet(
        PreActBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
    )


def PreActResNet152(num_classes=10):
    return PreActResNet(
        PreActBottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
    )
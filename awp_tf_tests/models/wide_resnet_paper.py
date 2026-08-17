import math
import tensorflow as tf


class KaimingFanOutInitializer(tf.keras.initializers.Initializer):
    """
    PyTorch equivalent of:

        nn.init.kaiming_normal_(
            weight,
            mode='fan_out',
            nonlinearity='relu'
        )

    For Conv2D weights in TensorFlow format:
        [kernel_h, kernel_w, in_channels, out_channels]
    """

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) < 2:
            raise ValueError(
                "KaimingFanOutInitializer requires at least 2 dimensions."
            )

        # TensorFlow Conv2D:
        # [kernel_h, kernel_w, in_channels, out_channels]
        receptive_field_size = 1

        if len(shape) > 2:
            receptive_field_size = math.prod(shape[:-2])

        fan_out = shape[-1] * receptive_field_size

        std = math.sqrt(2.0 / fan_out)

        return tf.random.normal(
            shape,
            mean=0.0,
            stddev=std,
            dtype=dtype,
        )

    def get_config(self):
        return {}


class BasicBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        in_planes,
        out_planes,
        stride,
        dropRate=0.0,
    ):
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.droprate = dropRate

        self.equalInOut = (in_planes == out_planes)

        # BN1
        self.bn1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            momentum=0.9,
        )

        # ReLU1
        self.relu1 = tf.keras.layers.ReLU()

        # Conv1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=KaimingFanOutInitializer(),
        )

        # BN2
        self.bn2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            momentum=0.9,
        )

        # ReLU2
        self.relu2 = tf.keras.layers.ReLU()

        # Conv2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=KaimingFanOutInitializer(),
        )

        # Shortcut only when dimensions change
        if not self.equalInOut:
            self.convShortcut = tf.keras.layers.Conv2D(
                filters=out_planes,
                kernel_size=1,
                strides=stride,
                padding="valid",
                use_bias=False,
                kernel_initializer=KaimingFanOutInitializer(),
            )
        else:
            self.convShortcut = None

        self.dropout = (
            tf.keras.layers.Dropout(dropRate)
            if dropRate > 0
            else None
        )

    def call(self, x, training=None):

        if not self.equalInOut:

            # PyTorch:
            #
            # x = self.relu1(self.bn1(x))
            #
            # IMPORTANT:
            # This changes x itself.
            x = self.bn1(x, training=training)
            x = self.relu1(x)

            out = self.conv1(x)

        else:

            # PyTorch:
            #
            # out = self.relu1(self.bn1(x))
            #
            # x remains unchanged for the shortcut.
            out = self.bn1(x, training=training)
            out = self.relu1(out)

            out = self.conv1(out)

        # PyTorch:
        #
        # out = self.relu2(self.bn2(out))
        out = self.bn2(out, training=training)
        out = self.relu2(out)

        if self.droprate > 0:
            out = self.dropout(out, training=training)

        # Second convolution
        out = self.conv2(out)

        # Residual addition
        if self.equalInOut:
            shortcut = x
        else:
            shortcut = self.convShortcut(x)

        return shortcut + out


class NetworkBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        nb_layers,
        in_planes,
        out_planes,
        block,
        stride,
        dropRate=0.0,
    ):
        super().__init__()

        self.layer = []

        for i in range(int(nb_layers)):

            current_in_planes = (
                in_planes
                if i == 0
                else out_planes
            )

            current_stride = (
                stride
                if i == 0
                else 1
            )

            self.layer.append(
                block(
                    current_in_planes,
                    out_planes,
                    current_stride,
                    dropRate,
                )
            )

    def call(self, x, training=None):

        for block in self.layer:
            x = block(x, training=training)

        return x


class WideResNet(tf.keras.Model):

    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        dropRate=0.0,
    ):
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError(
                "WideResNet depth should satisfy "
                "(depth - 4) % 6 == 0."
            )

        n = (depth - 4) // 6

        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        self.nChannels = nChannels[3]

        # Initial convolution
        self.conv1 = tf.keras.layers.Conv2D(
            filters=nChannels[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=KaimingFanOutInitializer(),
        )

        # First block
        self.block1 = NetworkBlock(
            n,
            nChannels[0],
            nChannels[1],
            BasicBlock,
            stride=1,
            dropRate=dropRate,
        )

        # Second block
        self.block2 = NetworkBlock(
            n,
            nChannels[1],
            nChannels[2],
            BasicBlock,
            stride=2,
            dropRate=dropRate,
        )

        # Third block
        self.block3 = NetworkBlock(
            n,
            nChannels[2],
            nChannels[3],
            BasicBlock,
            stride=2,
            dropRate=dropRate,
        )

        # Final BN + ReLU
        self.bn1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            momentum=0.9,
        )

        self.relu = tf.keras.layers.ReLU()

        # Classifier
        self.fc = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer="zeros",
        )

    def call(self, x, training=None):

        # Initial convolution
        out = self.conv1(x)

        # Network blocks
        out = self.block1(out, training=training)
        out = self.block2(out, training=training)
        out = self.block3(out, training=training)

        # Final BN + ReLU
        out = self.bn1(out, training=training)
        out = self.relu(out)

        # Global average pooling over 8x8
        out = tf.nn.avg_pool2d(
            out,
            ksize=8,
            strides=8,
            padding="VALID",
        )

        # PyTorch:
        #
        # out = out.view(-1, self.nChannels)
        #
        out = tf.reshape(
            out,
            [-1, self.nChannels],
        )

        # Classifier
        return self.fc(out)
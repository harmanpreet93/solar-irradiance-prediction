import typing
import datetime
from model_logging import get_logger
import tensorflow as tf
import numpy as np


class MainModel(tf.keras.Model):

    TRAINING_REQUIRED = True

    def __init__(
        self,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """
        Args:
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation)
            target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding any extra parameters that might be required by the user. These
                parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
                such a JSON file is completely optional, and this argument can be ignored if not needed.
        """
        super(MainModel, self).__init__()
        self.stations = stations
        self.target_time_offsets = target_time_offsets
        self.config = config
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Model start")

        nb_channels = self.config["nb_channels"]
        image_size_m = self.config["image_size_m"]
        image_size_n = self.config["image_size_n"]
        images_per_pred = self.config["images_per_prediction"]

        self.conv3d_1 = tf.keras.layers.Conv3D(
            filters=16,
            kernel_size=(6, 6, images_per_pred),
            input_shape=(image_size_m, image_size_n, images_per_pred, nb_channels),
            strides=(2, 2, 1)
        )
        self.pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.conv2d_3 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2)
        )
        self.pool_4 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.conv2d_5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(4, 4),
            strides=(1, 1)
        )  # Note: this acts as a fully-connected network (input and kernel are same dim)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_6 = tf.keras.layers.Dense(
            256,
            activation=tf.nn.relu
        )
        self.dense_7 = tf.keras.layers.Dense(
            len(self.target_time_offsets),
            activation=tf.nn.relu
        )

    def call(self, inputs):
        '''
        Defines the forward pass through our model
        '''
        images = inputs[0]
        # clearsky_GHIs = inputs[1]
        # true_GHIs = inputs[2]  # NOTE: True GHI is set to zero for formal evaluation
        # night_flags = inputs[3]

        assert not np.isnan(images).any()

        x = self.conv3d_1(images)
        x = tf.squeeze(x)
        x = self.pool_2(x)  # 1 px lost here; TODO: consider padding the tensor in future
        x = self.conv2d_3(x)
        x = self.pool_4(x)
        x = self.conv2d_5(x)
        x = self.flatten(x)
        x = self.dense_6(x)
        y = self.dense_7(x)

        assert not np.isnan(y).any()

        return y

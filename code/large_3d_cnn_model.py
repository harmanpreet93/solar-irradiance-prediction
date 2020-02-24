import typing
import datetime
import numpy as np
import tensorflow as tf
from model_logging import get_logger
from training_loop import k_to_true_ghi


class MainModel(tf.keras.Model):

    TRAINING_REQUIRED = True

    def __init__(
        self,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        return_ghi_only=False
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
        self.return_ghi_only = return_ghi_only
        self.max_k_ghi = config["max_k_ghi"]
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Model start")

        nb_channels = self.config["nb_channels"]
        image_size_m = self.config["image_size_m"]
        image_size_n = self.config["image_size_n"]
        input_seq_length = self.config["input_seq_length"]

        self.conv3d_1 = tf.keras.layers.Conv3D(
            filters=self.config["nb_feature_maps"],
            kernel_size=(input_seq_length, 7, 7),
            input_shape=(input_seq_length, image_size_m, image_size_n, nb_channels),
            padding="valid",
            strides=(1, 1, 1),
            activation=None
        )
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.pool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=(2 * self.config["nb_feature_maps"]),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv2d_2 = tf.keras.layers.Conv2D(
            filters=(2 * self.config["nb_feature_maps"]),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.conv2d_3 = tf.keras.layers.Conv2D(
            filters=(4 * self.config["nb_feature_maps"]),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.conv2d_4 = tf.keras.layers.Conv2D(
            filters=(4 * self.config["nb_feature_maps"]),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.conv2d_5 = tf.keras.layers.Conv2D(
            filters=(4 * self.config["nb_feature_maps"]),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn_6 = tf.keras.layers.BatchNormalization()
        self.pool_3 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            self.config["nb_dense_units"],
            activation=None
        )
        self.bn_7 = tf.keras.layers.BatchNormalization()
        self.dense_2 = tf.keras.layers.Dense(
            self.config["nb_dense_units"],
            activation=None
        )
        self.bn_8 = tf.keras.layers.BatchNormalization()
        # Output layer
        self.dense_3 = tf.keras.layers.Dense(
            len(self.target_time_offsets),
            activation=None
        )

    def call(self, inputs, training=False):
        '''
        Defines the forward pass through our model
        '''
        images = tf.squeeze(inputs[0])
        clearsky_GHIs = tf.squeeze(inputs[1])
        # true_GHIs = inputs[2]  # NOTE: True GHI is set to zero for formal evaluation
        # night_flags = inputs[3]
        station_id_onehot = inputs[4]

        # Refer to report for mean/std choices
        normalized_clearsky_GHIs = (clearsky_GHIs - 454.5) / 293.9

        # assert not np.isnan(images).any()

        x = self.conv3d_1(images)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.squeeze(x)
        x = self.pool_1(x)

        x = self.conv2d_1(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_2(x)
        x = self.bn_3(x, training=training)
        x = tf.nn.relu(x)

        x = self.pool_2(x)

        x = self.conv2d_3(x)
        x = self.bn_4(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_4(x)
        x = self.bn_5(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_5(x)
        x = self.bn_6(x, training=training)
        x = tf.nn.relu(x)

        x = self.pool_3(x)

        x = self.flatten(x)
        x = tf.concat((x, station_id_onehot, normalized_clearsky_GHIs), axis=1)

        x = self.dense_1(x)
        x = self.bn_7(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense_2(x)
        x = self.bn_8(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense_3(x)
        k = tf.nn.sigmoid(x)

        # assert not np.isnan(k).any()

        y = k_to_true_ghi(self.max_k_ghi, k, clearsky_GHIs)

        if self.return_ghi_only:
            return y

        return k, y

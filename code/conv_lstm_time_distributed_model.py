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

        self.conv3d_1 = tf.keras.layers.Conv2D(
            filters=self.config["nb_feature_maps"],
            kernel_size=(3, 3),
            input_shape=(image_size_m, image_size_n, nb_channels),
            strides=(2, 2),
            activation=tf.nn.relu
        )

        self.pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.conv2d_3 = tf.keras.layers.Conv2D(
            filters=self.config['nb_feature_maps'],
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=tf.nn.relu
        )
        self.pool_4 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        self.flatten = tf.keras.layers.Flatten()

        self.time1 = tf.keras.layers.TimeDistributed(self.conv3d_1, input_shape=(
            input_seq_length, image_size_m, image_size_n, nb_channels))
        self.time2 = tf.keras.layers.TimeDistributed(self.pool_2)
        self.time3 = tf.keras.layers.TimeDistributed(self.conv2d_3)
        self.time4 = tf.keras.layers.TimeDistributed(self.pool_4)
        self.time5 = tf.keras.layers.TimeDistributed(self.flatten)
        self.lstm_6 = tf.keras.layers.LSTM(units=128)
        self.dense_7 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense_8 = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)

    def call(self, inputs):
        '''
        Defines the forward pass through our model
        '''
        images = tf.squeeze(inputs[0])
        clearsky_GHIs = tf.squeeze(inputs[1])
        # true_GHIs = inputs[2]  # NOTE: True GHI is set to zero for formal evaluation
        # night_flags = inputs[3]
        station_id_onehot = (inputs[4])

        # assert not np.isnan(images).any()

        x = self.time1(images)
        x = self.time2(x)
        x = self.time3(x)
        x = self.time4(x)
        x = self.time5(x)
        x = self.lstm_6(x)
        x = tf.concat((x, station_id_onehot), axis=1)
        # x = self.dense_7(x)
        k = self.dense_8(x)

        assert not np.isnan(k).any()

        y = k_to_true_ghi(self.max_k_ghi, k, clearsky_GHIs)

        if self.return_ghi_only:
            return y

        return k, y

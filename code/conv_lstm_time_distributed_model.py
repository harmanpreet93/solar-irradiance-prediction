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

        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=self.config["nb_feature_maps"],
            kernel_size=(3, 3),
            input_shape=(image_size_m, image_size_n, nb_channels),
            strides=(1, 1), padding='same'
        )

        self.relu1 = tf.keras.layers.Activation(activation=tf.nn.relu)

        self.pool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        self.conv2d_3 = tf.keras.layers.Conv2D(
            filters=2 * self.config['nb_feature_maps'],
            kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        )

        self.relu2 = tf.keras.layers.Activation(activation=tf.nn.relu)

        self.pool_4 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        self.conv2d_5 = tf.keras.layers.Conv2D(
            filters=3 * self.config['nb_feature_maps'],
            kernel_size=(3, 3),
            strides=(1, 1), padding='same'
        )

        self.relu3 = tf.keras.layers.Activation(activation=tf.nn.relu)

        self.pool_6 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        self.conv2d_6 = tf.keras.layers.Conv2D(
            filters=4 * self.config['nb_feature_maps'],
            kernel_size=(3, 3),
            strides=(2, 2), padding='same'
        )

        self.relu4 = tf.keras.layers.Activation(activation=tf.nn.relu)

        self.pool_7 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=self.config["dropout_rate"])

        self.flatten_5 = tf.keras.layers.Flatten()

        self.time1 = tf.keras.layers.TimeDistributed(self.conv2d_1, input_shape=(
            input_seq_length, image_size_m, image_size_n, nb_channels))
        self.time2 = tf.keras.layers.TimeDistributed(self.batch_norm_1)
        self.time3 = tf.keras.layers.TimeDistributed(self.relu1)
        self.time4 = tf.keras.layers.TimeDistributed(self.pool_2)

        self.time1_2 = tf.keras.layers.TimeDistributed(self.conv2d_3)
        self.time2_2 = tf.keras.layers.TimeDistributed(self.batch_norm_2)
        self.time3_2 = tf.keras.layers.TimeDistributed(self.relu2)
        self.time4_2 = tf.keras.layers.TimeDistributed(self.pool_4)

        self.time1_3 = tf.keras.layers.TimeDistributed(self.conv2d_5)
        self.time2_3 = tf.keras.layers.TimeDistributed(self.batch_norm_3)
        self.time3_3 = tf.keras.layers.TimeDistributed(self.relu3)
        self.time4_3 = tf.keras.layers.TimeDistributed(self.pool_6)

        self.time1_4 = tf.keras.layers.TimeDistributed(self.conv2d_6)
        self.time2_4 = tf.keras.layers.TimeDistributed(self.batch_norm_4)
        self.time3_4 = tf.keras.layers.TimeDistributed(self.relu4)
        self.time4_4 = tf.keras.layers.TimeDistributed(self.pool_7)

        self.time5 = tf.keras.layers.TimeDistributed(self.flatten_5)

        self.lstm_6_1 = tf.keras.layers.LSTM(units=384, return_sequences=True, recurrent_activation=tf.nn.relu)
        self.lstm_6_2 = tf.keras.layers.LSTM(units=256, recurrent_activation=tf.nn.relu)
        self.dense_7 = tf.keras.layers.Dense(self.config["nb_dense_units"]/2, activation=tf.nn.relu)
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

        date_sin_cos_vector = (inputs[5])

        # Refer to report for mean/std choices
        normalized_clearsky_GHIs = (clearsky_GHIs - 454.5) / 293.9

        # assert not np.isnan(images).any()

        x = self.time1(images)
        x = self.time2(x)
        x = self.time3(x)
        x = self.time4(x)

        x = self.time1_2(x)
        x = self.time2_2(x)
        x = self.time3_2(x)
        x = self.time4_2(x)

        x = self.time1_3(x)
        x = self.time2_3(x)
        x = self.time3_3(x)
        x = self.time4_3(x)

        x = self.time1_4(x)
        x = self.time2_4(x)
        x = self.time3_4(x)
        x = self.time4_4(x)

        x = self.time5(x)

        # print(x.shape)
        x = self.lstm_6_1(x)
        # print(x.shape)
        x = self.lstm_6_2(x)
        # print(x.shape)
        x = tf.concat((x, station_id_onehot, date_sin_cos_vector, normalized_clearsky_GHIs), axis=1)
        # print(x.shape)

        x = self.dense_7(x)
        # print(x.shape)

        x = self.batch_norm_5(x)
        x = self.dropout1(x)
        k = self.dense_8(x)

        assert not np.isnan(k).any()

        y = k_to_true_ghi(self.max_k_ghi, k, clearsky_GHIs)

        if self.return_ghi_only:
            return y

        return k, y

import typing
import datetime
import logging
import tensorflow as tf


class MainModel(tf.keras.Model):

    def __init__(
        self,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """
        Args:
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
            target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding any extra parameters that might be required by the user. These
                parameters aZre loaded automatically if the user provided a JSON file in their submission. Submitting
                such a JSON file is completely optional, and this argument can be ignored if not needed.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Model start")
        super(MainModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(len(target_time_offsets), activation=tf.nn.softmax)

    def call(self, inputs):
        self.logger.debug("Model call")
        image = inputs[0]
        clearsky_GHIs = inputs[1]
        # x = self.dense1(self.flatten(image))
        # x1 = self.dense2(x)
        return clearsky_GHIs

import typing
import datetime
from model_logging import get_logger
import tensorflow as tf


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
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal
        )
        self.dense2 = tf.keras.layers.Dense(
            units=len(self.target_time_offsets),
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.RandomNormal
        )

    def call(self, inputs):
        '''
        Defines the forward pass through our model
        '''
        image = inputs[0]
        # clearsky_GHIs = inputs[1]
        true_GHIs = inputs[2]  # NOTE: True GHI is set to zero for formal evaluation
        x = self.dense1(self.flatten(image))
        x1 = self.dense2(x)
        return true_GHIs + x1  # This trains the NN to predict x1 = 0

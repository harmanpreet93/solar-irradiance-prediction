import datetime
import typing
import tensorflow as tf
import pandas as pd


class DataLoader():

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """
        Copy-paste from evaluator.py:
        Args:
            dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
                relevant timestamp values over the test period.
            target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
                The ordering of this list is important, as each element corresponds to a sequence of GHI values
                to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
                which are added to each timestamp (T=0) in this datetimes list.
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
            target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding extra parameters

        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        """
        super().__init__()
        self.target_datetimes = target_datetimes
        self.config = config
        self.data_loader = tf.data.Dataset.from_generator(
            self.dummy_data_generator, (tf.float32, tf.float32)
        )

    def dummy_data_generator(self):
        """
        Generate dummy data for the model, only for example purposes.
        """
        batch_size = self.config["batch_size"]
        image_dim = (64, 64)
        n_channels = 5
        output_seq_len = 4

        for i in range(0, len(self.target_datetimes), batch_size):
            batch_of_datetimes = self.target_datetimes[i:i+batch_size]
            samples = tf.random.uniform(shape=(
                len(batch_of_datetimes), image_dim[0], image_dim[1], n_channels
            ))
            targets = tf.zeros(shape=(
                len(batch_of_datetimes), output_seq_len
            ))
            # Remember that you do not have access to the targets.
            # Your dataloader should handle this accordingly.
            yield samples, targets

    def get_data_loader(self):
        return self.data_loader

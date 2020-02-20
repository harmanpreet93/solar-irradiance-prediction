import datetime
import typing
from model_logging import get_logger

import tensorflow as tf
import pandas as pd
import numpy as np
import h5py
import utils


def get_nighttime_flags(batch_of_datetimes):
    batch_size = len(batch_of_datetimes)
    # TODO: Return real nighttime flags; assume no nighttime values for now
    return np.zeros(shape=(batch_size, 4))


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
                to predict. By definition, the GHI values must be provided for the offsets given by
                ``target_time_offsets`` which are added to each timestamp (T=0) in this datetimes list.
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation)
            target_time_offsets: the list of time-deltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding extra parameters
        """
        self.dataframe = dataframe
        self.target_datetimes = target_datetimes
        self.stations = list(stations.keys())
        self.stations_lat_longs = stations
        self.config = config
        self.target_time_offsets = target_time_offsets
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Initialize start")
        self.test_station = self.stations[0]
        self.output_seq_len = len(self.target_time_offsets)
        self.input_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in self.config["input_time_offsets"]]
        self.data_loader = tf.data.Dataset.from_generator(
            self.data_generator_fn,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        ).batch(self.config["batch_size"]).prefetch(self.config["batch_size"])

    def get_ghi_values(self, batch_of_datetimes, station_id):
        batch_size = len(batch_of_datetimes)
        batch_of_clearsky_GHIs = np.zeros((batch_size, self.output_seq_len))
        batch_of_true_GHIs = np.zeros((batch_size, self.output_seq_len))

        for i, dt in enumerate(batch_of_datetimes):
            for j, time_offset in enumerate(self.target_time_offsets):
                dt_index = dt + time_offset
                batch_of_clearsky_GHIs[i, j] = self.dataframe.lookup([dt_index], [station_id + '_CLEARSKY_GHI'])
                # If the true GHI is not provided, return 0s instead:
                if station_id + "_GHI" in self.dataframe.columns:
                    batch_of_true_GHIs[i, j] = self.dataframe.lookup([dt_index], [station_id + '_GHI'])

        batch_of_true_GHIs = np.nan_to_num(batch_of_true_GHIs)  # TODO: We only convert nan to 0 for now

        return batch_of_true_GHIs, batch_of_clearsky_GHIs

    def get_image_data(self, batch_of_datetimes):
        nb_channels = self.config["nb_channels"]
        image_size_m = self.config["image_size_m"]
        image_size_n = self.config["image_size_n"]
        images_per_pred = self.config["images_per_prediction"]
        batch_size = len(batch_of_datetimes)
        # TODO: Not implemented yet, generate random data instead
        image = tf.random.uniform(
            shape=(batch_size, image_size_m, image_size_n, images_per_pred, nb_channels)
        )
        return image

    # def data_generator_fn(self):
    #     batch_size = self.config["batch_size"]
    #     for station_id in self.stations:
    #         for i in range(0, len(self.target_datetimes), batch_size):
    #             batch_of_datetimes = self.target_datetimes[i:(i + batch_size)]
    #             true_GHIs, clearsky_GHIs = self.get_ghi_values(batch_of_datetimes, station_id)
    #             images = self.get_image_data(batch_of_datetimes)
    #             night_flags = self.get_nighttime_flags(batch_of_datetimes)
    #
    #             # Remember that you do not have access to the targets.
    #             # Your dataloader should handle this accordingly.
    #             yield images, clearsky_GHIs, true_GHIs, night_flags, true_GHIs

    def normalize_images(self, x):
        return x

    def crop_images(self,
                    df: pd.DataFrame,
                    timestamps_from_history: list,
                    coordinates: typing.Dict[str, typing.Tuple],
                    window_size: float,
                    ):

        assert window_size < 42, f"window_size value of {window_size} is too big, please reduce it to 42 and lower"

        n_channels = 5
        image_crops = np.zeros(shape=(
            len(timestamps_from_history), window_size * 2, window_size * 2, n_channels
        ))

        for index, timestamp in enumerate(timestamps_from_history):
            # print("Timestamp: {}".format(timestamp))
            try:
                row = df.loc[timestamp]
            except:
                print("Timestamp {} not found, do something!".format(timestamp))
                print("Not considering this sequence while training")
                return None

            hdf5_path = row["hdf5_8bit_path"]
            # hdf5_path = '/project/cq-training-1/project1/data/hdf5v7_8bit/2015.01.01.0800.h5'
            # file_date = hdf5_path.split("/")[-1][:-3]
            # date of the file
            # file_date = "_".join(file_date.split('.'))
            hdf5_offset = row["hdf5_8bit_offset"]
            # print("harman ncdf_path ", row["ncdf_path"])

            for station_coordinates in coordinates.items():
                # retrieves station name and coordinates for each station
                station_name = station_coordinates[0]
                x_coord = station_coordinates[1][0]
                y_coord = station_coordinates[1][1]

                with h5py.File(hdf5_path, "r") as h5_data:
                    # normalize arrays and crop the station for each channel
                    ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
                    ch2_data = self.normalize_images(utils.fetch_hdf5_sample("ch2", h5_data, hdf5_offset))
                    ch3_data = self.normalize_images(utils.fetch_hdf5_sample("ch3", h5_data, hdf5_offset))
                    ch4_data = self.normalize_images(utils.fetch_hdf5_sample("ch4", h5_data, hdf5_offset))
                    ch6_data = self.normalize_images(utils.fetch_hdf5_sample("ch6", h5_data, hdf5_offset))

                    if ch1_data is None or ch2_data is None or ch3_data is None or ch4_data is None or ch6_data is None:
                        return None

                    ch1_crop = ch1_data[x_coord - window_size:x_coord + window_size,
                               y_coord - window_size:y_coord + window_size]
                    ch2_crop = ch2_data[x_coord - window_size:x_coord + window_size,
                               y_coord - window_size:y_coord + window_size]
                    ch3_crop = ch3_data[x_coord - window_size:x_coord + window_size,
                               y_coord - window_size:y_coord + window_size]
                    ch4_crop = ch4_data[x_coord - window_size:x_coord + window_size,
                               y_coord - window_size:y_coord + window_size]
                    ch6_crop = ch6_data[x_coord - window_size:x_coord + window_size,
                               y_coord - window_size:y_coord + window_size]
                    image_crops[index] = np.stack((ch1_crop, ch2_crop, ch3_crop, ch4_crop, ch6_crop), axis=-1)

                    # save the images as .h5 file, will need to specify path
                    # generate_images(img_crop, station_name, file_date, hdf5_offset)

        return image_crops

    def get_TrueGHIs(self, timestamp, station_id):
        trueGHIs = [0] * 4
        GHI_col = station_id + "_GHI"
        # if timestamp missing
        try:
            trueGHIs[0] = self.dataframe.loc[timestamp][GHI_col]  # T0_GHI
            trueGHIs[1] = self.dataframe.loc[timestamp + self.target_time_offsets[1]][GHI_col]  # T1_GHI
            trueGHIs[2] = self.dataframe.loc[timestamp + self.target_time_offsets[2]][GHI_col]  # T3_GHI
            trueGHIs[3] = self.dataframe.loc[timestamp + self.target_time_offsets[3]][GHI_col]  # T6_GHI
        except:
            return None

        return trueGHIs

    def get_ClearSkyGHIs(self, timestamp, station_id):
        clearSkyGHIs = [0] * 4
        clearSkyGHI_col = station_id + "_CLEARSKY_GHI"
        # if timestamp missing
        try:
            clearSkyGHIs[0] = self.dataframe.loc[timestamp][clearSkyGHI_col]
            clearSkyGHIs[1] = self.dataframe.loc[timestamp + self.target_time_offsets[1]][clearSkyGHI_col]
            clearSkyGHIs[2] = self.dataframe.loc[timestamp + self.target_time_offsets[2]][clearSkyGHI_col]
            clearSkyGHIs[3] = self.dataframe.loc[timestamp + self.target_time_offsets[3]][clearSkyGHI_col]
        except:
            return None

        return clearSkyGHIs

    def get_stations_coordinates(self,
                                 df: pd.DataFrame
                                 ) -> typing.Dict[str, typing.Tuple]:

        """
        :param datafram_path: str pointing to the dataframe .pkl file
        :param stations_lats_lons: dictionary of str -> (latitude, longitude) of the station(s)
        :return: dictionary of str -> (coord_x, coord_y) in the numpy array
        """
        # takes the first value for the hdf5 path
        # hdf5_path = df["hdf5_8bit_path"][2]
        hdf5_path = "/project/cq-training-1/project1/data/hdf5v7_8bit/2015.01.01.0800.h5"
        # print("**********************hdf5 path ", hdf5_path)

        with h5py.File(hdf5_path, 'r') as h5_data:
            # print("h5_data file ", h5_data)
            lats, lons = utils.fetch_hdf5_sample("lat", h5_data, 0), utils.fetch_hdf5_sample("lon", h5_data, 0)

        stations_coords = {}

        for region, lats_lons in self.stations_lat_longs.items():
            coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
            stations_coords[region] = coords

        return stations_coords

    def preprocess_and_filter_data(self, main_df):

        # make sure it sorted by its index
        main_df.sort_index(ascending=True, inplace=True)

        # replace nan by np.nan
        main_df.replace('nan', np.NaN, inplace=True)

        # dropping records without hdf5 files
        main_df.drop(main_df.loc[main_df['hdf5_8bit_path'].isnull()].index, inplace=True)

        # split the dataframe by stations
        BND_df = main_df.iloc[:, 0:9]
        TBL_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 9:13]], axis=1)
        DRA_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 13:17]], axis=1)
        FPK_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 17:21]], axis=1)
        GWN_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 21:25]], axis=1)
        PSU_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 25:29]], axis=1)
        SXF_df = pd.concat([main_df.iloc[:, 0:4], main_df.iloc[:, 29:33]], axis=1)

        # Save the station's data as a dictionary of dataframes
        stations_df = {
            'BND': BND_df,
            'TBL': TBL_df,
            'DRA': DRA_df,
            'FPK': FPK_df,
            'GWN': GWN_df,
            'PSU': PSU_df,
            'SXF': SXF_df
        }

        return stations_df

    def data_generator_fn(self):
        main_dataframe_copy = self.dataframe.copy()

        station_wise_dataframes = self.preprocess_and_filter_data(main_dataframe_copy)

        for station_id in self.stations:
            station_df = station_wise_dataframes[station_id]
            DAYTIME = np.str(station_id + '_DAYTIME')

            # get station coordinates, need to be called only once
            stations_coordinates = self.get_stations_coordinates(station_df)

            # sort based on datetime
            station_df.sort_index(ascending=True, inplace=True)

            # drop all night times as of now: to make solution simpler
            station_df.drop(station_df.loc[station_df[DAYTIME] == 0.0].index, inplace=True)

            for time_index, col in station_df[self.config["input_seq_length"]:].iterrows():
                # get past timestamps in range of input_seq_length
                timestamps_from_history = []
                for i in range(self.config["input_seq_length"]):
                    timestamps_from_history.append(time_index - self.input_time_offsets[i])

                true_GHIs = self.get_TrueGHIs(time_index, station_id)
                if true_GHIs is None:
                    continue

                clearsky_GHIs = self.get_ClearSkyGHIs(time_index, station_id)
                if clearsky_GHIs is None:
                    continue

                night_flags = np.zeros(4)

                # get cropped images for given timestamp
                # tensor of size (input_seq_length x C x W x H)
                images = self.crop_images(station_df, timestamps_from_history,
                                          {station_id: stations_coordinates[station_id]},
                                          window_size=self.config["image_size_m"] // 2)

                if images is None:
                    continue

                # print("Images size: ", images.shape)
                # print("GHIs ", true_GHIs, clearsky_GHIs)

                yield images, clearsky_GHIs, true_GHIs, night_flags, true_GHIs

    def get_data_loader(self):
        '''
        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        '''
        return self.data_loader

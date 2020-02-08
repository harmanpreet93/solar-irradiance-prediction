import utils
import numpy as np
import pandas as pd
import h5py
import json
import typing

from pathlib import Path

data_folder = Path("Documents/MILA/IFT6759/Solar-irradiance-Team08-IFT6759")

with open(data_folder / 'pre_process_cfg.json', 'r') as fd:
    pre_process_config = json.load(fd)

hdf5_path = pre_process_config["hdf5_path"]
dataframe_path = pre_process_config["dataframe_path"]
stations = pre_process_config["stations"]


def get_stations_coordinates(
        dataframe_path: str,
        stations_lats_lons: typing.Dict[str, typing.Tuple]
    ) -> typing.Dict[str, typing.Tuple]:

    """
    :param datafram_path: str pointing to the dataframe .pkl file
    :param stations_lats_lons: dictionnary of str -> (latitude, longitude) of the station(s)
    :return: dictionnary of str -> (coord_x, coord_y) in the numpy array 
    """
    df = pd.read_pickle(dataframe_path) if dataframe_path else None
    # takes the first non Nan or 'nan' value for the hdf5 path
    hdf5_path = df.loc[df["hdf5_8bit_path"].notnull() == True & (df["hdf5_8bit_path"]!= 'nan')]["hdf5_8bit_path"][0]
    with h5py.File(hdf5_path, 'r') as h5_data:
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, 0), utils.fetch_hdf5_sample("lon", h5_data, 0)
    stations_coords = {}
    for region, lats_lons in stations_lats_lons.items():
        coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
        stations_coords[region] = coords
    return stations_coords


def crop_images(
    dataframe_path: str,
    window_size: float
    ):
    """ 
    :param datafram_path: str pointing to the dataframe .pkl file
    :param window_size : float defining the pixel range of the crop centered at a station
    :return:
    """
    assert window_size < 42, f"window_size value of {window_size} is too big, please reduce it to 42 and lower"
    df = pd.read_pickle(dataframe_path) if dataframe_path else None
    df_copy = df.copy().replace(to_replace="nan",
                                value=np.NaN).dropna(subset=["hdf5_8bit_path"])
    coordinates = get_stations_coordinates(dataframe_path, stations)
    for index, row in df_copy.iterrows():
        hdf5_path = row["hdf5_8bit_path"]
        hdf5_offset = row["hdf5_8bit_offset"]
        for station_coordinates in coordinates.items():
            x_coord = station_coordinates[1][0]
            y_coord = station_coordinates[1][1]
            with h5py.File(hdf5_path, "r") as h5_data:
                ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
                ch2_data = utils.fetch_hdf5_sample("ch2", h5_data, hdf5_offset)
                ch3_data = utils.fetch_hdf5_sample("ch3", h5_data, hdf5_offset)
                ch4_data = utils.fetch_hdf5_sample("ch4", h5_data, hdf5_offset)
                ch6_data = utils.fetch_hdf5_sample("ch6", h5_data, hdf5_offset)

                ch1_crop = ch1_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch2_crop = ch2_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch3_crop = ch3_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch4_crop = ch4_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch6_crop = ch6_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                img_crop = np.stack((ch1_crop, ch2_crop, ch3_crop, ch4_crop, ch6_crop), axis=-1)
                # save images and output new dataframe

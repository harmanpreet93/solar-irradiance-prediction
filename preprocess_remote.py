import numpy as np
import pandas as pd
import h5py
import json
import typing
from code import utils

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

# need to modify this so that the normalization takes the min, max of each channel and not each image
def normalize_images(
    image : np.ndarray
):
    """ 
    :param image: image as an array
    :return: an array that is normalized between 0 and 1 
    """
    image = (image - np.min(image)) / np.ptp(image)
    return image


def generate_images(
    cropped_image : np.ndarray,
    station : str,
    file_date : str,
    offset : int
):
    """ 
    :param cropped_image: array of a cropped station
    :param station: string -> station that was cropped
    :param file_day: str -> corresponding date of the file that is cropped
    :param offset: the integer index (or offset) that corresponds to the position of the sample in the dataset
    This function will save the cropped images as .h5 file in the format of cropped_filedate_station_offset.h5
    """
    # Can specify the path here at the beginning of the argument. Note that w stands for create file, truncate if exists
    with h5py.File(f"cropped_{file_date}" + "_" + f"{station}" + "_" + f"{offset}.hdf5" , "w") as f:
        crop = f.create_dataset("images", data=cropped_image)
        # can add other keys if needed


def crop_images(
    dataframe_path: str,
    window_size: float
    ):
    """ 
    :param datafram_path: str pointing to the dataframe .pkl file
    :param window_size : float defining the pixel range of the crop centered at a station. One pixel is 16km^2
    :return:
    """
    assert window_size < 42, f"window_size value of {window_size} is too big, please reduce it to 42 and lower"
    df = pd.read_pickle(dataframe_path) if dataframe_path else None
    df_copy = df.copy().replace(to_replace="nan",
                                value=np.NaN).dropna(subset=["hdf5_8bit_path"])
    coordinates = get_stations_coordinates(dataframe_path, stations)
    for index, row in df_copy.iterrows():
        hdf5_path = row["hdf5_8bit_path"]
        file_date = hdf5_path.split("/")[-1]
        # date of the file
        file_date = "_".join(file_date.split('.'))[:-3]
        hdf5_offset = row["hdf5_8bit_offset"]
        for station_coordinates in coordinates.items():
            # retrieves station name and coordinates for each station
            station_name = station_coordinates[0]
            x_coord = station_coordinates[1][0]
            y_coord = station_coordinates[1][1]
            with h5py.File(hdf5_path, "r") as h5_data:
                # normalize arrays and crop the station for each channel
                ch1_data = normalize_images(utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset))
                ch2_data = normalize_images(utils.fetch_hdf5_sample("ch2", h5_data, hdf5_offset))
                ch3_data = normalize_images(utils.fetch_hdf5_sample("ch3", h5_data, hdf5_offset))
                ch4_data = normalize_images(utils.fetch_hdf5_sample("ch4", h5_data, hdf5_offset))
                ch6_data = normalize_images(utils.fetch_hdf5_sample("ch6", h5_data, hdf5_offset))

                ch1_crop = ch1_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch2_crop = ch2_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch3_crop = ch3_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch4_crop = ch4_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                ch6_crop = ch6_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
                img_crop = np.stack((ch1_crop, ch2_crop, ch3_crop, ch4_crop, ch6_crop), axis=-1)
                # save the images as .h5 file, will need to specify path
                generate_images(img_crop, station_name, file_date, hdf5_offset)

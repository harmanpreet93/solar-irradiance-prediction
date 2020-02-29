import argparse
import datetime
import typing
import pandas as pd
import numpy as np
import h5py
import utils
import os
import tqdm
import json
import multiprocessing


def get_stations_coordinates(stations) -> typing.Dict[str, typing.Tuple]:
    """
    :return: dictionnary of str -> (coord_x, coord_y) mapping station coordinates to pixel  
    """
    # takes one hdf5 path
    hdf5_path = "/project/cq-training-1/project1/data/hdf5v7_8bit/2015.01.01.0800.h5"

    with h5py.File(hdf5_path, 'r') as h5_data:
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, 0), utils.fetch_hdf5_sample("lon", h5_data, 0)

    stations_coords = {}

    for region, lats_lons in stations.items():
        coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
        stations_coords[region] = coords

    return stations_coords


def preprocess_dataframe(dataframe: pd.DataFrame,
                         stations: typing.Dict[str, typing.Tuple]
    ) -> pd.DataFrame:
    """
    :return: preprocessed pd.Dataframe 
    """
    main_df = dataframe.copy()

    # make sure it sorted by its index
    main_df.sort_index(ascending=True, inplace=True)

    # replace nan by np.nan (why??)
    main_df.replace('nan', np.NaN, inplace=True)

    # dropping records without hdf5 files
    main_df.drop(main_df.loc[main_df['hdf5_8bit_path'].isnull()].index, inplace=True)

    # dropping records without ncdf files
    main_df.drop(main_df.loc[main_df['ncdf_path'].isnull()].index, inplace=True)

    # drop all night times
    station_names = list(stations.keys())
    b = [s + "_DAYTIME" for s in station_names]

    main_df.drop(main_df[(main_df[b[0]] == 0.0)
                         & (main_df[b[1]] == 0.0)
                         & (main_df[b[2]] == 0.0)
                         & (main_df[b[3]] == 0.0)
                         & (main_df[b[4]] == 0.0)
                         & (main_df[b[5]] == 0.0)
                         & (main_df[b[6]] == 0.0)].index, inplace=True)

    # shuffle dataframe
    main_df = main_df.sample(frac=1)

    return main_df


def load_file(path, name):
    assert os.path.isfile(path), f"invalid {name} config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def save_image_and_batch(dir_path,
                         file_name,
                         image_data,
                         true_ghis_data,
                         clear_sky_ghis_data,
                         station_ids,
                         datetime_sequence,
                         night_time_flags):
    """
    This function saves images, true GHI, clearsky_GHI, night_flags, and stations_ids in .h5 file  
    """
    file_name = file_name + ".hdf5"
    path = os.path.join(dir_path, file_name)
    with h5py.File(path, 'w') as f:
        f.create_dataset("images", shape=image_data.shape, dtype=np.float32, data=image_data)
        f.create_dataset("GHI", shape=true_ghis_data.shape, dtype=np.float32, data=true_ghis_data)
        f.create_dataset("clearsky_GHI", shape=clear_sky_ghis_data.shape, dtype=np.float32, data=clear_sky_ghis_data)
        f.create_dataset("night_flags", shape=night_time_flags.shape, dtype=np.float32, data=night_time_flags)
        station_ids = [n.encode("ascii", "ignore") for n in station_ids]
        f.create_dataset("station_id", shape=(len(station_ids), 1), dtype='S10', data=station_ids)

        # save station specific date in h5 file:  you can get other timestamps by adding/subtracting to this timestamp
        datetime_sequence = [str(n).encode("ascii", "ignore") for n in datetime_sequence]
        f.create_dataset("datetime_sequence", shape=(len(datetime_sequence), 1), dtype='S100', data=datetime_sequence)


def get_channels(hdf5_path, hdf5_offset):
    """
    :return: all the channels for a particular offset
    """
    with h5py.File(hdf5_path, "r") as h5_data:
        ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
        ch2_data = utils.fetch_hdf5_sample("ch2", h5_data, hdf5_offset)
        ch3_data = utils.fetch_hdf5_sample("ch3", h5_data, hdf5_offset)
        ch4_data = utils.fetch_hdf5_sample("ch4", h5_data, hdf5_offset)
        ch6_data = utils.fetch_hdf5_sample("ch6", h5_data, hdf5_offset)
    return ch1_data, ch2_data, ch3_data, ch4_data, ch6_data


def get_TrueGHIs(dataframe, target_time_offsets, timestamp, station_id, is_eval=False):
    """ 
    :return: list of true GHIs for T0, T1, T3, T6
    """
    if not is_eval:
        # check if its night at this station
        DAYTIME_col = station_id + "_DAYTIME"
        # return None if T0 time isn't daytime, we don't want to train on such sequences
        if dataframe.loc[timestamp][DAYTIME_col] == 0.0:
            return None

    trueGHIs = [0] * 4
    GHI_col = station_id + "_GHI"
    clearSkyGHI_col = station_id + "_CLEARSKY_GHI"

    # if timestamp missing
    # using clearsky value to interpolate
    try:
        trueGHIs[0] = dataframe.loc[timestamp][GHI_col]  # T0_GHI
    except Exception as _:
        try:
            trueGHIs[0] = dataframe.loc[timestamp + target_time_offsets[0]][clearSkyGHI_col]
        except Exception as _:
            trueGHIs[0] = 0

    try:
        trueGHIs[1] = dataframe.loc[timestamp + target_time_offsets[1]][GHI_col]  # T1_GHI
    except Exception as _:
        try:
            trueGHIs[1] = dataframe.loc[timestamp + target_time_offsets[1]][clearSkyGHI_col]
        except Exception as _:
            pass

    try:
        trueGHIs[2] = dataframe.loc[timestamp + target_time_offsets[2]][GHI_col]  # T3_GHI
    except Exception as _:
        try:
            trueGHIs[2] = dataframe.loc[timestamp + target_time_offsets[2]][clearSkyGHI_col]
        except Exception as _:
            pass

    try:
        trueGHIs[3] = dataframe.loc[timestamp + target_time_offsets[3]][GHI_col]  # T6_GHI
    except Exception as _:
        try:
            trueGHIs[3] = dataframe.loc[timestamp + target_time_offsets[3]][clearSkyGHI_col]
        except Exception as _:
            pass

    # if we still encounter NaN, handle it
    if np.any(np.isnan(trueGHIs)):
        return None

    return trueGHIs


def get_ClearSkyGHIs(dataframe, target_time_offsets, timestamp, station_id, is_eval=False):
    """
    :return: list of clear sky GHI's for T0, T1, T3, T6 
    """
    if not is_eval:
        # check if its night at this station
        DAYTIME_col = station_id + "_DAYTIME"
        # return None if T0 time isn't daytime, we don't want to train on such sequences
        if dataframe.loc[timestamp][DAYTIME_col] == 0.0:
            return None

    clearSkyGHIs = [0] * 4
    clearSkyGHI_col = station_id + "_CLEARSKY_GHI"
    # if timestamp missing
    try:
        clearSkyGHIs[0] = dataframe.loc[timestamp][clearSkyGHI_col]  # T0_GHI
    except Exception as _:
        try:
            clearSkyGHIs[0] = dataframe.loc[timestamp + target_time_offsets[0]][clearSkyGHI_col]
        except Exception as _:
            clearSkyGHIs[0] = 0

    try:
        clearSkyGHIs[1] = dataframe.loc[timestamp + target_time_offsets[1]][clearSkyGHI_col]  # T1_GHI
    except Exception as _:
        try:
            clearSkyGHIs[1] = dataframe.loc[timestamp + target_time_offsets[0]][clearSkyGHI_col]
        except Exception as _:
            clearSkyGHIs[1] = 1

    try:
        clearSkyGHIs[2] = dataframe.loc[timestamp + target_time_offsets[2]][clearSkyGHI_col]  # T3_GHI
    except Exception as _:
        try:
            clearSkyGHIs[2] = dataframe.loc[timestamp + target_time_offsets[1]][clearSkyGHI_col]
        except Exception as _:
            clearSkyGHIs[2] = 1

    try:
        clearSkyGHIs[3] = dataframe.loc[timestamp + target_time_offsets[3]][clearSkyGHI_col]  # T6_GHI
    except Exception as _:
        try:
            clearSkyGHIs[3] = dataframe.loc[timestamp + target_time_offsets[2]][clearSkyGHI_col]
        except Exception as _:
            clearSkyGHIs[3] = 1

    # if we still encounter NaN, handle it
    if np.any(np.isnan(clearSkyGHIs)):
        return None

    return clearSkyGHIs


def get_night_time_flags(dataframe, target_time_offsets, timestamp, station_id, is_eval=False):
    """
    :return: list of daytime flags for T0, T1, T3, T6
    """
    # check if its night at this station
    DAYTIME_col = station_id + "_DAYTIME"

    if not is_eval:
        # return None if T0 time isn't daytime, we don't want to train on such sequences
        if dataframe.loc[timestamp][DAYTIME_col] == 0.0:
            return None

    night_time_flags = [1.0] * 4
    # if timestamp missing
    try:
        night_time_flags[0] = dataframe.loc[timestamp][DAYTIME_col]  # T0_GHI
    except Exception as _:
        night_time_flags[0] = 1.0
    try:
        night_time_flags[1] = dataframe.loc[timestamp + target_time_offsets[1]][DAYTIME_col]  # T1_GHI
    except Exception as _:
        night_time_flags[1] = 1.0

    try:
        night_time_flags[2] = dataframe.loc[timestamp + target_time_offsets[2]][DAYTIME_col]  # T3_GHI
    except Exception as _:
        night_time_flags[2] = 1.0

    try:
        night_time_flags[3] = dataframe.loc[timestamp + target_time_offsets[3]][DAYTIME_col]  # T6_GHI
    except Exception as _:
        night_time_flags[3] = 1.0

    # if we still encounter NaN, handle it
    if np.any(np.isnan(night_time_flags)):
        return None

    return night_time_flags


def get_station_specific_time(timestamp, station_id, time_zone_mapping):
    return timestamp - time_zone_mapping[station_id]


def normalize_images(images):
    """ 
    Standardize the images with mean 0 and variance 1
    """
    means = [0.30, 272.52, 236.94, 261.47, 247.28]
    stds = [0.218, 13.66, 6.49, 15.91, 11.15]

    for channel, u, sig in zip(range(5), means, stds):
        images[:, :, channel] = (images[:, :, channel] - u) / sig
    return images


def crop_images(df,
                big_df,
                timestamps_from_history,
                target_time_offsets,
                coordinates,
                window_size,
                time_zone_mapping,
                is_eval):
    """ 
    :return: multiple arrays corresponding to cropped images, true GHIs, clearsky GHIs,
    station IDs, T0 timestamp, nighttime flags
    """
    assert window_size < 42, f"window_size value of {window_size} is too big, please reduce it to 42 and lower"

    image_crops_for_stations = []
    true_ghis_for_station = []
    clearSky_ghis_for_station = []
    station_ids = []
    considered_timestamps = []
    night_time_flags_for_station = []

    for index, timestamp in enumerate(timestamps_from_history):
        try:
            row = df.loc[timestamp]
        except Exception as _:
            if index == 0:
                print("Timestamp {} not found for station {}, Not considering this sequence! \n".format(timestamp,
                                                                                                        coordinates))
                return None, None, None, None
            else:
                # use T0 image if missing
                row = df.loc[timestamps_from_history[0]]

        hdf5_path = row["hdf5_8bit_path"]
        hdf5_offset = row["hdf5_8bit_offset"]

        channels_data = get_channels(hdf5_path, hdf5_offset)

        image_crops_per_stations = []

        # get cropped images for stations
        for _, station_coordinates in enumerate(coordinates.items()):

            station_id = station_coordinates[0]
            DAYTIME_col = station_id + "_DAYTIME"
            # check if T0 time isn't daytime, we don't want to train on such sequences for that particular station
            if not is_eval and df.loc[timestamps_from_history[0]][DAYTIME_col] == 0.0:
                continue

            x_coord = station_coordinates[1][0]
            y_coord = station_coordinates[1][1]

            ch1_crop = channels_data[0][x_coord - window_size:x_coord + window_size,
                                        y_coord - window_size:y_coord + window_size]
            ch2_crop = channels_data[1][x_coord - window_size:x_coord + window_size,
                                        y_coord - window_size:y_coord + window_size]
            ch3_crop = channels_data[2][x_coord - window_size:x_coord + window_size,
                                        y_coord - window_size:y_coord + window_size]
            ch4_crop = channels_data[3][x_coord - window_size:x_coord + window_size,
                                        y_coord - window_size:y_coord + window_size]
            ch6_crop = channels_data[4][x_coord - window_size:x_coord + window_size,
                                        y_coord - window_size:y_coord + window_size]

            cropped_img = np.stack((ch1_crop,
                                    ch2_crop,
                                    ch3_crop,
                                    ch4_crop,
                                    ch6_crop), axis=-1)

            cropped_img = normalize_images(cropped_img)

            cropped_img = np.expand_dims(cropped_img, axis=0)

            image_crops_per_stations.append(cropped_img)

            # get true GHIs only for first timestamp - T0
            if index == 0:
                trueGHIs = get_TrueGHIs(big_df, target_time_offsets, timestamp, station_id, is_eval)
                clearSkyGHIs = get_ClearSkyGHIs(big_df, target_time_offsets, timestamp, station_id, is_eval)
                night_time_flags = get_night_time_flags(big_df, target_time_offsets, timestamp, station_id, is_eval)

                # we don't need GHIs during evaluation
                if is_eval:
                    trueGHIs = [0] * 4

                if trueGHIs is None:
                    # setup dummy GHIs for now!
                    trueGHIs = np.ones(len(target_time_offsets))

                if clearSkyGHIs is None:
                    clearSkyGHIs = np.ones(len(target_time_offsets))

                if night_time_flags is None:
                    night_time_flags = np.ones(len(target_time_offsets))

                true_ghis_for_station.append(trueGHIs)
                clearSky_ghis_for_station.append(clearSkyGHIs)
                # append station id if everything works well till this point
                station_ids.append(station_id)
                timestamp_as_per_station_timezone = get_station_specific_time(timestamp, station_id, time_zone_mapping)
                considered_timestamps.append(timestamp_as_per_station_timezone)
                night_time_flags_for_station.append(night_time_flags)

        image_crops_per_stations = np.array(image_crops_per_stations)

        # image_crops_for_stations.append(image_crops_per_stations)
        if len(image_crops_for_stations) == 0:
            image_crops_for_stations = image_crops_per_stations
        else:
            image_crops_for_stations = np.concatenate((image_crops_for_stations, image_crops_per_stations), axis=1)

    return np.array(image_crops_for_stations), np.array(true_ghis_for_station), np.array(
        clearSky_ghis_for_station), np.array(station_ids), np.array(considered_timestamps), np.array(
        night_time_flags_for_station)


def save_batches(main_df, dataframe, stations_coordinates, user_config, train_config, save_dir_path, start_index,
                 end_index, mini_batch_size, is_eval=False):
    """ 
    Helper function for create_and_save_batches function
    """
    input_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in user_config["input_time_offsets"]]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in train_config["target_time_offsets"]]
    time_zone_mapping = {k: pd.Timedelta(d).to_pytimedelta() for k, d in user_config["time_zone_mapping"].items()}

    window_size = user_config["image_size_m"] // 2

    concat_images = np.array([])
    target_trueGHIs = np.array([])
    target_clearSkyGHIs = np.array([])
    target_timestamps = np.array([])
    target_station_ids = np.array([])
    target_night_time_flags = np.array([])
    index = 0
    batch_counter = start_index

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for time_index, _ in tqdm.tqdm(main_df[start_index:end_index].iterrows()):
        # get past timestamps in range of input_seq_length
        timestamps_from_history = []
        for i in range(user_config["input_seq_length"]):
            timestamps_from_history.append(time_index - input_time_offsets[i])

        images, trueGHIs, clearSkyGHIs, station_ids, timestamps, night_time_flags = \
            crop_images(main_df,
                        dataframe,
                        timestamps_from_history,
                        target_time_offsets,
                        stations_coordinates,
                        window_size,
                        time_zone_mapping,
                        is_eval)

        if images is None:
            # print("No image found for timestamp {}".format(time_index))
            # print("No target GHIs found for timestamp {}".format(time_index))
            continue

        if trueGHIs is None:
            continue

        if len(concat_images) == 0:
            concat_images = images
            target_trueGHIs = trueGHIs
            target_clearSkyGHIs = clearSkyGHIs
            target_timestamps = timestamps
            target_station_ids = station_ids
            target_night_time_flags = night_time_flags
        else:
            concat_images = np.append(concat_images, images, axis=0)
            target_trueGHIs = np.append(target_trueGHIs, trueGHIs, axis=0)
            target_clearSkyGHIs = np.append(target_clearSkyGHIs, clearSkyGHIs, axis=0)
            target_timestamps = np.append(target_timestamps, timestamps, axis=0)
            target_station_ids = np.append(target_station_ids, station_ids, axis=0)
            target_night_time_flags = np.append(target_night_time_flags, night_time_flags, axis=0)

        index += images.shape[0]

        # save h5py file here
        if index >= mini_batch_size:
            assert concat_images.shape[0] == target_trueGHIs.shape[0]
            assert concat_images.shape[0] == target_clearSkyGHIs.shape[0]
            assert concat_images.shape[0] == target_timestamps.shape[0]
            assert concat_images.shape[0] == target_station_ids.shape[0]
            assert concat_images.shape[0] == target_night_time_flags.shape[0]

            batch_counter += 1
            file_name = "batch_val_" + str(batch_counter).zfill(4)

            save_image_and_batch(save_dir_path, file_name,
                                 concat_images[:mini_batch_size],
                                 target_trueGHIs[:mini_batch_size],
                                 target_clearSkyGHIs[:mini_batch_size],
                                 target_station_ids[:mini_batch_size],
                                 target_timestamps[:mini_batch_size],
                                 target_night_time_flags[:mini_batch_size])
            # print(target_timestamps)
            # break

            concat_images = np.array([])
            target_trueGHIs = np.array([])
            target_clearSkyGHIs = np.array([])
            target_station_ids = np.array([])
            target_timestamps = np.array([])
            target_night_time_flags = np.array([])

            index = 0


def fill_zero_for_night_ghi_nans(df, column_name, stations):
    """ 
    Helper function that fills NaNs GHIs with 0
    """
    for station in tqdm.tqdm(stations, total=7):
        col = station + "_" + column_name
        daytime_col = station + "_DAYTIME"
        df[col] = df.apply(
            lambda row: 0.0 if (np.isnan(row[col]) and (row[daytime_col] == 0.0)) else row[col],
            axis=1
        )
    return df


# interpolate GHI NaNs to zero
def interpolate_ghi_nans(df, column_name, stations):
    """
    Helper function that linearly interpolates NaN GHIs from both side 
    """
    for station in tqdm.tqdm(stations, total=7):
        col = station + "_" + column_name
        df[col] = df[col].interpolate(limit_direction='both')
    return df


def handle_ghi_nans(old_df, handle_true_ghi=True, handle_clearsky_ghis=True):
    """ 
    :return: pd.DataFrame with linearly interpolated previously missing GHIs
    """
    stations = ["BND", "TBL", "DRA", "FPK", "GWN", "PSU", "SXF"]
    df = old_df.copy()
    df.replace('nan', np.NaN, inplace=True)

    df.sort_index(ascending=True, inplace=True)

    if handle_true_ghi:
        print("Setting night GHI NaNs to 0")
        df = fill_zero_for_night_ghi_nans(df, 'GHI', stations)
        print("Interpolating GHIs")
        df = interpolate_ghi_nans(df, 'GHI', stations)
        print("Done")

    if handle_clearsky_ghis:
        print("Setting night clearsky GHI NaNs to 0")
        df = fill_zero_for_night_ghi_nans(df, 'CLEARSKY_GHI', stations)
        print("Interpolating clearsky GHIs")
        df = interpolate_ghi_nans(df, 'CLEARSKY_GHI', stations)
        print("Done")

    return df


def create_and_save_batches(
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        is_eval=False
) -> None:
    """ 
    This function create and save batches of images and other features used for training in .h5 file
    """
    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    stations = admin_config["stations"]
    # get station coordinates, need to be called only once, or save its value in config file
    stations_coordinates = get_stations_coordinates(stations)

    if is_eval:
        print("Evaluating model:  {}".format(user_config["target_model"]))
        print("\nPreprocessing data...")
        # replace nan by np.nan
        dataframe.replace('nan', np.NaN, inplace=True)
        # dropping records without hdf5 files
        dataframe.drop(dataframe.loc[dataframe['hdf5_8bit_path'].isnull()].index, inplace=True)
        # dropping records without ncdf files
        dataframe.drop(dataframe.loc[dataframe['ncdf_path'].isnull()].index, inplace=True)

        print("Filtering dataframe based on start and end bound dates...")
        filtered_dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

        target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
        assert target_datetimes and all([d in filtered_dataframe.index for d in target_datetimes])

        filtered_dataframe_ = filtered_dataframe.loc[target_datetimes]

        args_array = []
        for station, _ in stations.items():
            val_file_path = os.path.join(user_config['val_data_folder'], str(station))
            mini_batch_size = 1
            args = (
                filtered_dataframe_, dataframe, {station: stations_coordinates[station]}, user_config,
                admin_config, val_file_path, 0,
                len(filtered_dataframe_) + 1, mini_batch_size, is_eval)
            args_array.append(args)

        p = multiprocessing.Pool(4)
        print("Saving batches now...")
        p.starmap(save_batches, args_array)
        print("Done")

        # for station, _ in stations.items():
        #     print("Creating batches for station: {}".format(station))
        #
        #     val_file_path = os.path.join(user_config['val_data_folder'], str(station))
        #     mini_batch_size = 1
        #     save_batches(filtered_dataframe_, dataframe, {station: stations_coordinates[station]}, user_config,
        #                  admin_config, val_file_path, 0,
        #                  len(filtered_dataframe_) + 1, mini_batch_size, is_eval)
        #
        #     print("Done \n")

    else:
        dataframe = handle_ghi_nans(dataframe, handle_true_ghi=True, handle_clearsky_ghis=True)

        train_dataframe = dataframe.loc['2010-01-01':'2015-01-01']
        val_dataframe = dataframe.loc['2015-01-01':'2015-12-31']

        train_dataframe = preprocess_dataframe(train_dataframe, stations)
        val_dataframe = preprocess_dataframe(val_dataframe, stations)

        my_train_args = []
        mini_batch_size = user_config['mini_batch_size']
        step_size = 500
        train_file_path = "/project/cq-training-1/project1/teams/team08/data/train_crops_seq_3_harman"
        # renaming it to save in the same folder as train
        val_file_path = "/project/cq-training-1/project1/teams/team08/data/train_crops_seq_3_harman"

        for i in range(0, len(train_dataframe) + 1000, step_size):
            args = (
                train_dataframe, dataframe, stations_coordinates, user_config, admin_config, train_file_path, int(i),
                int(i) + step_size, mini_batch_size)
            my_train_args.append(args)

        my_val_args = []
        for i in range(0, len(val_dataframe) + 1000, step_size):
            args = (val_dataframe, dataframe, stations_coordinates, user_config, admin_config, val_file_path, int(i),
                    int(i) + step_size, mini_batch_size)
            my_val_args.append(args)

        p = multiprocessing.Pool(4)
        print("Saving batches now...")
        p.starmap(save_batches, my_val_args)
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters",
                        default="../val_cfg_local.json")
    parser.add_argument("-u", "--user_cfg_path", type=str,
                        help="path to the JSON config file used to store user model/dataloader parameters",
                        default="eval_user_cfg_lstm.json")

    args = parser.parse_args()

    create_and_save_batches(
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        is_eval=False
    )

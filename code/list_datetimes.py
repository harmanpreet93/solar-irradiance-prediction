import pandas as pd
import numpy as np
import json
import datetime
import os
import h5py
import argparse
import typing


def clean_df(df):
    df.replace(to_replace="nan", value=np.NaN, inplace=True)


image_type = "hdf5_8bit"


def get_datetimes_from_df(df, station_id):
    daytime = df[station_id + "_DAYTIME"] == 1.0
    valid_ghi = ~df[station_id + "_GHI"].isna()
    valid_datetime = daytime & valid_ghi
    return df.loc[valid_datetime].index


# Check that all channels used are available
def check_all_chan_ok(h5_data, hdf5_offset):
    for channel in ["ch1", "ch2", "ch3", "ch4", "ch6"]:
        dataset_lut_name = channel + "_LUT"
        # This definition is used by the util package
        if dataset_lut_name in h5_data and h5_data[dataset_lut_name][hdf5_offset] == -1:
            return False
    return True


# Check if the file and offset exist on disk.
def check_datetime_data_ok(hdf5_path, hdf5_offset):
    if os.path.isfile(hdf5_path):
        with h5py.File(hdf5_path, "r") as h5_data:
            data_ok = check_all_chan_ok(h5_data, hdf5_offset)
            if data_ok:
                return True
    return False


def load_file(path, name):
    assert os.path.isfile(path), f"invalid {name} config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def load_files(user_config_path):
    user_config = load_file(user_config_path, "user")
    return user_config


def load_df(user_config):
    dataframe_path = user_config["dataframe_path"]
    return pd.read_pickle(dataframe_path)


def check_if_data_available(df, candidate_datetimes):

    datetimes = []

    for datetime in candidate_datetimes:
        # For each retained datetime Lookup in the dataframe which file to check.
        # target_datetime = df.loc[datetime, ["hdf5_8bit_path", "hdf5_8bit_offset"]]

        # hdf5_path = target_datetime["hdf5_8bit_path"]
        # hdf5_offset = target_datetime["hdf5_8bit_offset"]

        # # Temporary local implementation for df lookup; uses Jan 2015
        # hdf5_path = "data/hdf5v7_8bit_Jan_2015/2015.01.01.0800.h5"

        # data_ok = check_datetime_data_ok(hdf5_path, hdf5_offset)
        # if data_ok:
        datetimes += [datetime]

    return datetimes


def get_datetimes_with_past_im_avail(candidate_datetimes):

    nb_images_for_training = 3
    timedeltas = pd.timedelta_range(start='-15 min', periods=(nb_images_for_training-1), freq='-15 min')
    print(candidate_datetimes[2])
    print(candidate_datetimes[2] + timedeltas)

    final_datetimes = []
    i = 0
    for candidate_datetime in candidate_datetimes:
        i += 1
        if i == 10:
            break
        timedelta_ok = [elem in candidate_datetimes for elem in (candidate_datetime + timedeltas)]
        if all(timedelta_ok):
            final_datetimes += [candidate_datetime]

    return final_datetimes


def write_datetimes_to_json_cfg_file(train_timestamps, val_timestamps, user_config, station_id):
    train_datetimes = [ts.isoformat() for ts in train_timestamps]
    val_datetimes = [ts.isoformat() for ts in val_timestamps]

    stations_dict = {
        "BND": [40.05192, -88.37309, 230],
        "TBL": [40.12498, -105.23680, 1689],
        "DRA": [36.62373, -116.01947, 1007],
        "FPK": [48.30783, -105.10170, 634],
        "GWN": [34.25470, -89.87290, 98],
        "PSU": [40.72012, -77.93085, 376],
        "SXF": [43.73403, -96.62328, 473]
    }

    train_output = {
        "dataframe_path": "data/catalog.helios.public.20100101-20160101.pkl",
        "start_bound": user_config["start_bound_train"],
        "end_bound": user_config["end_bound_train"],
        "stations": {station_id: stations_dict[station_id]},
        "target_time_offsets": [
            "P0DT0H0M0S",
            "P0DT1H0M0S",
            "P0DT3H0M0S",
            "P0DT6H0M0S"
        ]}

    train_output["target_datetimes"] = train_datetimes

    with open('train_cfg_{}.json'.format(station_id), 'w') as outfile:
        json.dump(train_output, outfile, indent=2)

    val_output = train_output
    val_output["target_datetimes"] = val_datetimes
    val_output["start_bound"] = user_config["start_bound_val"]
    val_output["end_bound"] = user_config["end_bound_val"]

    with open('val_cfg_{}.json'.format(station_id), 'w') as outfile:
        json.dump(val_output, outfile, indent=2)


def clip_datetimes(user_config, processed_datetimes):
    start_tr = datetime.datetime.fromisoformat(user_config["start_bound_train"])
    end_tr = datetime.datetime.fromisoformat(user_config["end_bound_train"])
    start_val = datetime.datetime.fromisoformat(user_config["start_bound_val"])
    end_val = datetime.datetime.fromisoformat(user_config["end_bound_val"])

    train_datetimes = [dt for dt in processed_datetimes if start_tr < dt and dt < end_tr]
    val_datetimes = [dt for dt in processed_datetimes if start_val < dt and dt < end_val]

    return train_datetimes, val_datetimes


def main(
        user_config_path: typing.Optional[typing.AnyStr],
) -> None:
    user_config = load_files(user_config_path)

    df = load_df(user_config)
    clean_df(df)
    station_id = "BND"
    candidate_datetimes = get_datetimes_from_df(df, station_id)

    datetime_availability_df = check_if_data_available(df, candidate_datetimes)

    processed_datetimes = get_datetimes_with_past_im_avail(datetime_availability_df)

    train_datetimes, val_datetimes = clip_datetimes(user_config, processed_datetimes)

    write_datetimes_to_json_cfg_file(train_datetimes, val_datetimes, user_config, station_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        user_config_path=args.user_cfg_path,
    )

import json
import pandas as pd
import os
from data_loader import DataLoader
import argparse
import typing
import datetime


def load_file(path, name):
    assert os.path.isfile(path), f"invalid {name} config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def load_files(user_config_path, train_config_path, val_config_path):
    user_config = load_file(user_config_path, "user")
    train_config = load_file(train_config_path, "training")
    val_config = load_file(val_config_path, "validation")

    dataframe_path = train_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    return user_config, train_config, val_config, dataframe


def clip_dataframe(dataframe, train_config):
    if "start_bound" in train_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(train_config["start_bound"])]
    if "end_bound" in train_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(train_config["end_bound"])]
    return dataframe


def get_targets(dataframe, config):
    datetimes = [datetime.datetime.fromisoformat(d) for d in config["target_datetimes"]]
    assert datetimes and all([d in dataframe.index for d in datetimes])

    stations = config["stations"]
    time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in config["target_time_offsets"]]

    return datetimes, stations, time_offsets


def main(
        train_config_path: typing.AnyStr,
        val_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    user_config, train_config, val_config, dataframe = \
        load_files(user_config_path, train_config_path, val_config_path)

    dataframe = \
        clip_dataframe(dataframe, train_config)

    tr_datetimes, tr_stations, tr_time_offsets = \
        get_targets(dataframe, train_config)

    val_datetimes, val_stations, val_time_offsets = \
        get_targets(dataframe, val_config)

    Train_DL = DataLoader(dataframe, tr_datetimes, tr_stations, tr_time_offsets, user_config)
    Val_DL = DataLoader(dataframe, val_datetimes, val_stations, val_time_offsets, user_config)
    Train_DL.data_generator_fn()
    Val_DL.data_generator_fn()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_cfg_path", type=str,
                        help="path to the JSON config file used to store training set parameters")
    parser.add_argument("val_cfg_path", type=str,
                        help="path to the JSON config file used to store validation set parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        train_config_path=args.train_cfg_path,
        val_config_path=args.val_cfg_path,
        user_config_path=args.user_cfg_path,
    )

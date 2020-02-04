import argparse
import datetime
import json
import os
import typing

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def loss_fn(y_true, y_pred):
    # TODO
    return tf.reduce_mean(tf.square(y_true - y_pred))


def generate_predictions(
    data_loader: tf.data.Dataset,
    model: tf.keras.Model,
    pred_count: int
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    optimizer = tf.keras.optimizers.Adam()
    with tqdm.tqdm("generating predictions", total=pred_count) as pbar:
        for epoch in range(10):
            cumulative_loss = 0.0
            for iter_idx, minibatch in enumerate(data_loader):
                assert isinstance(minibatch, tuple) and len(minibatch) >= 2
                with tf.GradientTape() as tape:
                    predictions = model(minibatch[:-1], training=True)
                    targets = minibatch[-1]
                    assert predictions.ndim == 2, "prediction tensor shape should be BATCH x SEQ_LENGTH"
                    loss = loss_fn(y_true=targets, y_pred=predictions)
                    cumulative_loss += loss
                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            print("Loss = " + str(loss.numpy()))
            pbar.update(len(predictions))


def generate_all_predictions(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    # we will create one data loader per station to make sure we avoid mixups in predictions
    for station_idx, station_name in enumerate(target_stations):
        # usually, we would create a single data loader for all stations, but we just want to avoid trouble...
        stations = {station_name: target_stations[station_name]}
        print(f"preparing data loader & model for station '{station_name}' ({station_idx + 1}/{len(target_stations)})")

        from data_loader import DataLoader
        DL = DataLoader(dataframe, target_datetimes, stations, target_time_offsets, user_config)
        data_loader = DL.get_data_loader()

        from main_model import MainModel
        model = MainModel(stations, target_time_offsets, user_config)

        generate_predictions(data_loader, model, pred_count=len(target_datetimes))


def parse_gt_ghi_values(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(station_ghis.iloc[station_ghis.index.get_loc(index)])
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] > 0)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def load_files(user_config_path, admin_config_path):
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

    return user_config, admin_config, dataframe


def clip_dataframe(dataframe, admin_config):
    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]
    return dataframe


def get_targets(dataframe, admin_config):
    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
    return target_datetimes, target_stations, target_time_offsets


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config, admin_config, dataframe = \
        load_files(user_config_path, admin_config_path)

    dataframe = \
        clip_dataframe(dataframe, admin_config)

    target_datetimes, target_stations, target_time_offsets = \
        get_targets(dataframe, admin_config)

    generate_all_predictions(target_stations, target_datetimes, target_time_offsets, dataframe, user_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
    )

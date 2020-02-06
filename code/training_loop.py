import argparse
import datetime
import json
import os
import typing

from data_loader import DataLoader
from main_model import MainModel
from model_logging import get_logger

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def train(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""

    DL = DataLoader(dataframe, target_datetimes, target_stations, target_time_offsets, user_config)
    data_loader = DL.get_data_loader()
    model = MainModel(target_stations, target_time_offsets, user_config)
    logger = get_logger()

    # set hyper-parameters
    n_epoch = user_config["n_epoch"]
    learning_rate = user_config["learning_rate"]
    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # MSE loss: as it is a regression problem
    # TODO: Better to use RMSE, as that's what we report in evaluation
    loss_fn = tf.keras.losses.mean_squared_error
    # training starts here
    # TODO: Add tensorboard logging
    with tqdm.tqdm("training", total=n_epoch) as pbar:
        for epoch in range(n_epoch):
            cumulative_loss = 0.0
            for minibatch in data_loader:
                assert isinstance(minibatch, tuple) and len(minibatch) >= 2
                with tf.GradientTape() as tape:
                    predictions = model(minibatch[:-1], training=True)
                    targets = minibatch[-1]
                    assert predictions.ndim == 2, "prediction tensor shape should be BATCH x SEQ_LENGTH"
                    loss = tf.reduce_mean(loss_fn(y_true=targets, y_pred=predictions))
                    cumulative_loss += loss
                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            logger.debug("Epoch {0}/{1}, Loss = {2}".format(epoch + 1, n_epoch, cumulative_loss.numpy()))
            pbar.update(1)

    # save model weights
    model.save_weights("model/my_model", save_format="tf")

    # return predictions from the model
    return np.array([])


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

    train(target_stations, target_datetimes, target_time_offsets, dataframe, user_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
    )

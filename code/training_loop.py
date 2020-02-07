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


def compute_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


def train(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        training_datetimes: typing.List[datetime.datetime],
        validation_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
):
    """Trains and saves the model to file"""

    Train_DL = DataLoader(dataframe, training_datetimes, target_stations, target_time_offsets, user_config)
    Val_DL = DataLoader(dataframe, validation_datetimes, target_stations, target_time_offsets, user_config)

    train_data_loader = Train_DL.get_data_loader()
    val_data_loader = Val_DL.get_data_loader()

    model = MainModel(target_stations, target_time_offsets, user_config)
    logger = get_logger()

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]
    nb_training_samples = len(training_datetimes)
    nb_validation_samples = len(validation_datetimes)
    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # MSE loss: as it is a regression problem
    # TODO: Better to use RMSE, as that's what we report in evaluation
    loss_fn = tf.keras.losses.mean_squared_error
    # training starts here
    # TODO: Add tensorboard logging
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        for epoch in range(nb_epoch):

            # Train the model using the training set for one epoch
            cumulative_train_loss = 0.0
            cumulative_train_rmse = 0.0
            for minibatch in train_data_loader:
                with tf.GradientTape() as tape:
                    predictions = model(minibatch[:-1], training=True)
                    loss = tf.reduce_mean(loss_fn(y_true=minibatch[-1], y_pred=predictions))
                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                cumulative_train_loss += loss
                cumulative_train_rmse += compute_rmse(y_true=minibatch[-1], y_pred=predictions)
            cumulative_train_loss /= nb_training_samples
            cumulative_train_rmse /= nb_training_samples

            # Evaluate model performance on the validation set after training for one epoch
            cumulative_val_loss = 0.0
            cumulative_val_rmse = 0.0
            for minibatch in val_data_loader:
                predictions = model(minibatch[:-1])
                loss = tf.reduce_mean(loss_fn(y_true=minibatch[-1], y_pred=predictions))
                cumulative_val_loss += loss
                cumulative_val_rmse += compute_rmse(y_true=minibatch[-1], y_pred=predictions)
            cumulative_val_loss /= nb_validation_samples
            cumulative_val_rmse /= nb_validation_samples

            logger.debug(
                "Epoch {0}/{1}, Train RMSE = {2}, Val RMSE = {3}"
                .format(epoch + 1, nb_epoch, cumulative_train_rmse.numpy(), cumulative_val_rmse.numpy())
            )
            pbar.update(1)

    # save model weights to file
    model.save_weights("model/my_model", save_format="tf")


def load_files(user_config_path, train_config_path, val_config_path):
    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(train_config_path), f"invalid training config file: {train_config_path}"
    with open(train_config_path, "r") as fd:
        train_config = json.load(fd)

    assert os.path.isfile(val_config_path), f"invalid validation config file: {val_config_path}"
    with open(val_config_path, "r") as fd:
        val_config = json.load(fd)

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


def get_targets(dataframe, train_config, val_config):
    train_target_datetimes = [datetime.datetime.fromisoformat(d) for d in train_config["target_datetimes"]]
    val_target_datetimes = [datetime.datetime.fromisoformat(d) for d in val_config["target_datetimes"]]
    assert train_target_datetimes and all([d in dataframe.index for d in train_target_datetimes])
    assert val_target_datetimes and all([d in dataframe.index for d in val_target_datetimes])

    target_stations = train_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in train_config["target_time_offsets"]]

    return train_target_datetimes, val_target_datetimes, target_stations, target_time_offsets


def main(
        train_config_path: typing.AnyStr,
        val_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config, train_config, val_config, dataframe = \
        load_files(user_config_path, train_config_path, val_config_path)

    dataframe = \
        clip_dataframe(dataframe, train_config)

    training_datetimes, validation_datetimes, target_stations, target_time_offsets = \
        get_targets(dataframe, train_config, val_config)

    train(target_stations, training_datetimes, validation_datetimes, target_time_offsets, dataframe, user_config)


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

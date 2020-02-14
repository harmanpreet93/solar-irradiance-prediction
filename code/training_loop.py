import argparse
import datetime
import json
import os
import typing
import sys

from data_loader import DataLoader
from model_logging import get_logger, get_summary_writer

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm

logger = get_logger()


def do_code_profiling(function):
    def wrapper(*args, **kwargs):
        if args[-1]["code_profiling_enabled"]:
            import cProfile
            import pstats
            profile = cProfile.Profile()
            profile.enable()

            x = function(*args, **kwargs)

            profile.disable()
            profile.dump_stats("log/profiling_results.prof")
            with open("log/profiling_results.txt", "w") as f:
                ps = pstats.Stats("log/profiling_results.prof", stream=f)
                ps.sort_stats('cumulative')
                ps.print_stats()
            return x
        else:
            return function(*args, **kwargs)
    return wrapper


def mask_nighttime_predictions(y_pred, y_true, night_flag):
    day_flag = 1.0 - night_flag
    masked_y_pred = tf.multiply(y_pred, day_flag) + tf.multiply(y_true, night_flag)
    return masked_y_pred


def train_step(model, optimizer, loss_fn, x_train, y_train):
    with tf.GradientTape() as tape:
        y_pred = model(x_train, training=True)
        y_pred = mask_nighttime_predictions(y_pred, y_train, x_train[3])
        loss = loss_fn(y_train, y_pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss, y_train, y_pred


def test_step(model, loss_fn, x_test, y_test):
    y_pred = model(x_test)
    loss = loss_fn(y_test, y_pred)
    return loss, y_test, y_pred


@do_code_profiling
def train(
        MainModel,
        tr_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        val_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        tr_datetimes: typing.List[datetime.datetime],
        val_datetimes: typing.List[datetime.datetime],
        tr_time_offsets: typing.List[datetime.timedelta],
        val_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any]
):
    """Trains and saves the model to file"""

    # Import the training and validation data loaders, import the model
    Train_DL = DataLoader(dataframe, tr_datetimes, tr_stations, tr_time_offsets, user_config)
    Val_DL = DataLoader(dataframe, val_datetimes, val_stations, val_time_offsets, user_config)
    train_data_loader = Train_DL.get_data_loader()
    val_data_loader = Val_DL.get_data_loader()
    model = MainModel(tr_stations, tr_time_offsets, user_config)

    # Set up tensorboard logging
    train_summary_writer, test_summary_writer = get_summary_writer()

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]

    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Objective/Loss function: MSE Loss
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Metrics to track:
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    train_rmse = tf.keras.metrics.RootMeanSquaredError()
    test_rmse = tf.keras.metrics.RootMeanSquaredError()

    # training starts here
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        for epoch in range(nb_epoch):

            # Train the model using the training set for one epoch
            for minibatch in train_data_loader:
                loss, y_train, y_pred = train_step(
                    model,
                    optimizer,
                    loss_fn,
                    x_train=minibatch[:-1],
                    y_train=minibatch[-1]
                )
                train_loss(loss)
                train_rmse(y_train, y_pred)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('rmse', train_rmse.result(), step=epoch)

            # Evaluate model performance on the validation set after training for one epoch
            for minibatch in val_data_loader:
                loss, y_test, y_pred = test_step(
                    model,
                    loss_fn,
                    x_test=minibatch[:-1],
                    y_test=minibatch[-1]
                )
                test_loss(loss)
                test_rmse(y_test, y_pred)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('rmse', test_rmse.result(), step=epoch)

            logger.debug(
                "Epoch {0}/{1}, Train Loss = {2}, Val Loss = {3}"
                .format(epoch + 1, nb_epoch, train_loss.result(), test_loss.result())
            )

            # Reset metrics every epoch
            train_loss.reset_states()
            train_rmse.reset_states()
            test_loss.reset_states()
            test_rmse.reset_states()

            pbar.update(1)

    # save model weights to file
    model.save_weights("model/my_model", save_format="tf")


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


def select_model(user_config):
    if user_config["target_model"] == "truth_predictor_model":
        from truth_predictor_model import MainModel
    elif user_config["target_model"] == "clearsky_model":
        from clearsky_model import MainModel
    elif user_config["target_model"] == "3d_cnn_model":
        from cnn_3d_model import MainModel
    else:
        raise Exception("Unknown model")

    return MainModel


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

    MainModel = select_model(user_config)

    if MainModel.TRAINING_REQUIRED:
        train(
            MainModel,
            tr_stations,
            val_stations,
            tr_datetimes,
            val_datetimes,
            tr_time_offsets,
            val_time_offsets,
            dataframe,
            user_config
        )
    else:
        logger.warning("Model not trained; Model doesn't require training")


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
    logger.info(str(sys.argv))
    args = parse_args()
    main(
        train_config_path=args.train_cfg_path,
        val_config_path=args.val_cfg_path,
        user_config_path=args.user_cfg_path,
    )

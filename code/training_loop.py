import argparse
import datetime
import json
import os
import typing
import sys

from data_loader import DataLoader
from model_logging import get_logger

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm

logger = get_logger()


def compute_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


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
    Train_DL = DataLoader(dataframe, tr_datetimes, tr_stations, tr_time_offsets, user_config)
    Val_DL = DataLoader(dataframe, val_datetimes, val_stations, val_time_offsets, user_config)
    train_data_loader = Train_DL.get_data_loader()
    val_data_loader = Val_DL.get_data_loader()

    nb_training_samples = len(tr_datetimes)
    nb_validation_samples = len(val_datetimes)

    model = MainModel(tr_stations, tr_time_offsets, user_config)

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]

    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # RMSE loss: as it is a regression problem
    loss_fn = compute_rmse

    # training starts here
    # TODO: Add tensorboard logging
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        for epoch in range(nb_epoch):

            # Train the model using the training set for one epoch
            cumulative_train_loss = 0.0
            for minibatch in train_data_loader:
                with tf.GradientTape() as tape:
                    predictions = model(minibatch[:-1], training=True)
                    loss = loss_fn(y_true=minibatch[-1], y_pred=predictions)
                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                cumulative_train_loss += loss
            cumulative_train_loss /= nb_training_samples

            # Evaluate model performance on the validation set after training for one epoch
            cumulative_val_loss = 0.0
            for minibatch in val_data_loader:
                predictions = model(minibatch[:-1])
                cumulative_val_loss += loss_fn(y_true=minibatch[-1], y_pred=predictions)
            cumulative_val_loss /= nb_validation_samples

            logger.debug(
                "Epoch {0}/{1}, Train Loss = {2}, Val Loss = {3}"
                .format(epoch + 1, nb_epoch, cumulative_train_loss.numpy(), cumulative_val_loss.numpy())
            )
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

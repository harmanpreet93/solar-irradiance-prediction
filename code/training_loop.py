import tqdm
import typing
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from data_loader import DataLoader
from model_logging import get_logger, get_summary_writer, do_code_profiling

logger = get_logger()


def k_to_true_ghi(max_k_ghi, k, clearsky_ghi):
    true_ghi = tf.math.multiply_no_nan(k * max_k_ghi, clearsky_ghi)
    return true_ghi


def ghi_to_k(max_k_ghi, true_ghi, clearsky_ghi):
    true_ghi = tf.maximum(true_ghi, 0.0)
    k = tf.math.divide_no_nan(true_ghi, clearsky_ghi * max_k_ghi)
    # Clip too large and small k values
    k = tf.minimum(k, 1.0)
    k = tf.maximum(k, 0.0)
    return k


def mask_nighttime_predictions(*args, night_flag):
    day_flag = tf.logical_not(night_flag)
    weight = tf.reduce_sum(tf.cast(day_flag, tf.float32))
    outputs = []
    for arg in args:
        outputs += [tf.boolean_mask(tensor=arg, mask=day_flag)]
    return outputs + [weight]


def train_step(model, optimizer, loss_fn, max_k_ghi, x_train, y_train):
    k_train = ghi_to_k(max_k_ghi, true_ghi=y_train, clearsky_ghi=x_train[1])
    with tf.GradientTape() as tape:
        k_pred, y_pred = model(x_train, training=True)
        k_pred, k_train, y_pred, y_train, weight = \
            mask_nighttime_predictions(k_pred, k_train, y_pred, y_train, night_flag=x_train[3])
        loss = loss_fn(k_train, k_pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss, y_train, y_pred, weight


def test_step(model, loss_fn, max_k_ghi, x_test, y_test):
    k_test = ghi_to_k(max_k_ghi, true_ghi=y_test, clearsky_ghi=x_test[1])
    k_pred, y_pred = model(x_test)
    y_pred, y_test, k_pred, k_test, weight = \
        mask_nighttime_predictions(y_pred, y_test, k_pred, k_test, night_flag=x_test[3])
    loss = loss_fn(k_test, k_pred)
    return loss, y_test, y_pred, weight


def manage_model_checkpoints(optimizer, model, user_config):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './model/tf_ckpts', max_to_keep=3)

    if user_config["ignore_checkpoints"]:
        print("Model checkpoints ignored; Initializing from scratch.")
        early_stop_metric = np.inf
        np.save(user_config["model_info"], [early_stop_metric])
    else:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored model from {}".format(manager.latest_checkpoint))
            early_stop_metric = np.load(user_config["model_info"])[0]
        else:
            print("No checkpoint found; Initializing from scratch.")
            early_stop_metric = np.inf

    start_epoch = ckpt.step.numpy()

    return manager, ckpt, early_stop_metric, start_epoch


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

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]
    max_k_ghi = user_config["max_k_ghi"]

    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Objective/Loss function: MSE Loss
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Set up tensorboard metric logging
    train_summary_writer, test_summary_writer = get_summary_writer()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    train_rmse = tf.keras.metrics.RootMeanSquaredError()
    test_rmse = tf.keras.metrics.RootMeanSquaredError()

    # Checkpoint management (for model save/restore)
    manager, ckpt, early_stop_metric, start_epoch = manage_model_checkpoints(optimizer, model, user_config)

    # training starts here
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        pbar.update(start_epoch)
        for epoch in range(start_epoch, nb_epoch):

            # Train the model using the training set for one epoch
            for minibatch in train_data_loader:
                loss, y_train, y_pred, weight = train_step(
                    model,
                    optimizer,
                    loss_fn,
                    max_k_ghi,
                    x_train=minibatch[:-1],
                    y_train=minibatch[-1]
                )
                train_loss(loss, sample_weight=weight)
                train_rmse(y_train, y_pred, sample_weight=weight)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('rmse', train_rmse.result(), step=epoch)

            # Evaluate model performance on the validation set after training for one epoch
            for minibatch in val_data_loader:
                loss, y_test, y_pred, weight = test_step(
                    model,
                    loss_fn,
                    max_k_ghi,
                    x_test=minibatch[:-1],
                    y_test=minibatch[-1]
                )
                test_loss(loss, sample_weight=weight)
                test_rmse(y_test, y_pred, sample_weight=weight)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('rmse', test_rmse.result(), step=epoch)

            # Create a model checkpoint after each epoch
            ckpt.step.assign_add(1)
            save_path = manager.save()
            logger.debug("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))

            # Save the best model
            if test_loss.result() < early_stop_metric:
                early_stop_metric = test_loss.result()
                model.save_weights("model/my_model", save_format="tf")
                np.save(user_config["model_info"], [early_stop_metric.numpy()])

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

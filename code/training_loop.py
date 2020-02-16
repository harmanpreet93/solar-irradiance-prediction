import tqdm
import typing
import datetime
import pandas as pd
import tensorflow as tf
from data_loader import DataLoader
from model_logging import get_logger, get_summary_writer, do_code_profiling

logger = get_logger()
MAX_K_GHI = 1.2


def k_to_true_ghi(k, clearsky_ghi):
    # Clip too large and small k values
    k = tf.minimum(k, MAX_K_GHI)
    k = tf.maximum(k, 0.0)
    true_ghi = tf.math.multiply_no_nan(k, clearsky_ghi)
    return true_ghi


def ghi_to_k(true_ghi, clearsky_ghi):
    k = tf.math.divide_no_nan(true_ghi, clearsky_ghi)
    # Clip too large and small k values
    k = tf.minimum(k, MAX_K_GHI)
    k = tf.maximum(k, 0.0)
    return k


def mask_nighttime_predictions(y_pred, y_true, night_flag):
    day_flag = tf.logical_not(night_flag)
    y_pred = tf.boolean_mask(tensor=y_pred, mask=day_flag)
    y_true = tf.boolean_mask(tensor=y_true, mask=day_flag)
    weight = tf.reduce_sum(tf.cast(day_flag, tf.float32))
    return y_pred, y_true, weight


def train_step(model, optimizer, loss_fn, x_train, y_train):
    k_train = ghi_to_k(true_ghi=y_train, clearsky_ghi=x_train[1])
    with tf.GradientTape() as tape:
        k_pred = model(x_train, predict_k=True, training=True)
        y_pred = ghi_to_k(true_ghi=k_pred, clearsky_ghi=x_train[1])
        k_pred, k_train, weight = mask_nighttime_predictions(k_pred, k_train, x_train[3])
        loss = loss_fn(k_train, k_pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss, y_train, y_pred, weight


def test_step(model, loss_fn, x_test, y_test):
    y_pred = model(x_test, predict_k=False)
    y_pred, y_test, weight = mask_nighttime_predictions(y_pred, y_test, x_test[3])
    loss = loss_fn(y_test, y_pred)
    return loss, y_test, y_pred, weight


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
                loss, y_train, y_pred, weight = train_step(
                    model,
                    optimizer,
                    loss_fn,
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
                    x_test=minibatch[:-1],
                    y_test=minibatch[-1]
                )
                test_loss(loss, sample_weight=weight)
                test_rmse(y_test, y_pred, sample_weight=weight)

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

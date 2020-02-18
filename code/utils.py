import datetime
import json
import math
import os
import typing
import warnings

import cv2 as cv
import h5py
import lz4.frame
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm


def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""
    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0
    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def get_label_html_color_code(idx):
    """Returns the PASCAL VOC HTML color code for a given label index."""
    color_array = get_label_color_mapping(idx)
    return f"#{color_array[0]:02X}{color_array[1]:02X}{color_array[2]:02X}"


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible BGR array.
    The reason why we flip the channel order (RGB->BGR) is for OpenCV compatibility. Feel free to
    edit this function if you wish to use it with another display library.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf[..., ::-1]


def compress_array(
        array: np.ndarray,
        compr_type: typing.Optional[str] = "auto",
) -> bytes:
    """Compresses the provided numpy array according to a predetermined strategy.
    If ``compr_type`` is 'auto', the best strategy will be automatically selected based on the input
    array type. If ``compr_type`` is an empty string (or ``None``), no compression will be applied.
    """
    assert compr_type is None or compr_type in ["lz4", "float16+lz4", "uint8+jpg",
                                                "uint8+jp2", "uint16+jp2", "auto", ""], \
        f"unrecognized compression strategy '{compr_type}'"
    if compr_type is None or not compr_type:
        return array.tobytes()
    if compr_type == "lz4":
        return lz4.frame.compress(array.tobytes())
    if compr_type == "float16+lz4":
        assert np.issubdtype(array.dtype, np.floating), "no reason to cast to float16 is not float32/64"
        return lz4.frame.compress(array.astype(np.float16).tobytes())
    if compr_type == "uint8+jpg":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jpg compression via tensorflow requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8, "jpg compression requires uint8 array"
        return tf.io.encode_jpeg(array).numpy()
    if compr_type == "uint8+jp2" or compr_type == "uint16+jp2":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jp2 compression via opencv requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8 or array.dtype == np.uint16, "jp2 compression requires uint8/16 array"
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        retval, buffer = cv.imencode(".jp2", array)
        assert retval, "JPEG2000 encoding failed"
        return buffer.tobytes()
    # could also add uint16 png/tiff via opencv...
    if compr_type == "auto":
        # we cheat for auto-decompression by prefixing the strategy in the bytecode
        if array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)):
            if array.dtype == np.uint8:
                return b"uint8+jpg" + compress_array(array, compr_type="uint8+jpg")
            if array.dtype == np.uint16:
                return b"uint16+jp2" + compress_array(array, compr_type="uint16+jp2")
        return b"lz4" + compress_array(array, compr_type="lz4")


def decompress_array(
        buffer: typing.Union[bytes, np.ndarray],
        compr_type: typing.Optional[str] = "auto",
        dtype: typing.Optional[typing.Any] = None,
        shape: typing.Optional[typing.Union[typing.List, typing.Tuple]] = None,
) -> np.ndarray:
    """Decompresses the provided numpy array according to a predetermined strategy.
    If ``compr_type`` is 'auto', the correct strategy will be automatically selected based on the array's
    bytecode prefix. If ``compr_type`` is an empty string (or ``None``), no decompression will be applied.
    This function can optionally convert and reshape the decompressed array, if needed.
    """
    compr_types = ["lz4", "float16+lz4", "uint8+jpg", "uint8+jp2", "uint16+jp2"]
    assert compr_type is None or compr_type in compr_types or compr_type in ["", "auto"], \
        f"unrecognized compression strategy '{compr_type}'"
    assert isinstance(buffer, bytes) or buffer.dtype == np.uint8, "invalid raw data buffer type"
    if isinstance(buffer, np.ndarray):
        buffer = buffer.tobytes()
    if compr_type == "lz4" or compr_type == "float16+lz4":
        buffer = lz4.frame.decompress(buffer)
    if compr_type == "uint8+jpg":
        # tf.io.decode_jpeg often segfaults when initializing parallel pipelines, let's avoid it...
        # buffer = tf.io.decode_jpeg(buffer).numpy()
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type.endswith("+jp2"):
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type == "auto":
        decompr_buffer = None
        for compr_code in compr_types:
            if buffer.startswith(compr_code.encode("ascii")):
                decompr_buffer = decompress_array(buffer[len(compr_code):], compr_type=compr_code,
                                                  dtype=dtype, shape=shape)
                break
        assert decompr_buffer is not None, "missing auto-decompression code in buffer"
        buffer = decompr_buffer
    array = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array


def fetch_hdf5_sample(
        dataset_name: str,
        reader: h5py.File,
        sample_idx: int,
) -> typing.Any:
    """Decodes and returns a single sample from an HDF5 dataset.
    Args:
        dataset_name: name of the HDF5 dataset to fetch the sample from using the reader. In the context of
            the GHI prediction project, this may be for example an imagery channel name (e.g. "ch1").
        reader: an HDF5 archive reader obtained via ``h5py.File(...)`` which can be used for dataset indexing.
        sample_idx: the integer index (or offset) that corresponds to the position of the sample in the dataset.
    Returns:
        The sample. This function will automatically decompress the sample if it was compressed. It the sample is
        unavailable because the input was originally masked, the function will return ``None``. The sample itself
        may be a scalar or a numpy array.
    """
    dataset_lut_name = dataset_name + "_LUT"
    if dataset_lut_name in reader:
        sample_idx = reader[dataset_lut_name][sample_idx]
        if sample_idx == -1:
            return None  # unavailable
    dataset = reader[dataset_name]
    if "compr_type" not in dataset.attrs:
        # must have been compressed directly (or as a scalar); return raw output
        return dataset[sample_idx]
    compr_type, orig_dtype, orig_shape = dataset.attrs["compr_type"], None, None
    if "orig_dtype" in dataset.attrs:
        orig_dtype = dataset.attrs["orig_dtype"]
    if "orig_shape" in dataset.attrs:
        orig_shape = dataset.attrs["orig_shape"]
    if "force_cvt_uint8" in dataset.attrs and dataset.attrs["force_cvt_uint8"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint8, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 255) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    elif "force_cvt_uint16" in dataset.attrs and dataset.attrs["force_cvt_uint16"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint16, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 65535) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    else:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=orig_dtype, shape=orig_shape)
    return array


def viz_hdf5_imagery(
        hdf5_path: str,
        channels: typing.List[str],
        dataframe_path: typing.Optional[str] = None,
        stations: typing.Optional[typing.Dict[str, typing.Tuple]] = None,
        copy_last_if_missing: bool = True,
) -> None:
    """Displays a looping visualization of the imagery channels saved in an HDF5 file.
    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The visualization can also be paused by pressing the space bar.
    """
    assert os.path.isfile(hdf5_path), f"invalid hdf5 path: {hdf5_path}"
    assert channels, "list of channels must not be empty"
    with h5py.File(hdf5_path, "r") as h5_data:
        global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
        global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
        archive_lut_size = global_end_idx - global_start_idx
        global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
        lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
        # will only display GHI values if dataframe is available
        stations_data = {}
        if stations:
            df = pd.read_pickle(dataframe_path) if dataframe_path else None
            # assume lats/lons stay identical throughout all frames; just pick the first available arrays
            idx, lats, lons = 0, None, None
            while (lats is None or lons is None) and idx < archive_lut_size:
                lats, lons = fetch_hdf5_sample("lat", h5_data, idx), fetch_hdf5_sample("lon", h5_data, idx)
            assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
            for reg, coords in tqdm.tqdm(stations.items(), desc="preparing stations data"):
                station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
                station_data = {"coords": station_coords}
                if dataframe_path:
                    station_data["ghi"] = [df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
                    station_data["csky"] = [df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
                stations_data[reg] = station_data
        raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500, 3), dtype=np.uint8)
        for channel_idx, channel_name in tqdm.tqdm(enumerate(channels), desc="preparing img data", total=len(channels)):
            assert channel_name in h5_data, f"missing channel: {channels}"
            norm_min = h5_data[channel_name].attrs.get("orig_min", None)
            norm_max = h5_data[channel_name].attrs.get("orig_max", None)
            channel_data = [fetch_hdf5_sample(channel_name, h5_data, idx) for idx in range(archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                for station_idx, (station_name, station) in enumerate(stations_data.items()):
                    station_color = get_label_color_mapping(station_idx + 1).tolist()[::-1]
                    array = cv.circle(array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
                raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
                last_valid_array_idx = array_idx
    plot_data = None
    if stations and dataframe_path:
        plot_data = preplot_live_ghi_curves(
            stations=stations, stations_data=stations_data,
            window_start=global_start_time,
            window_end=global_start_time + datetime.timedelta(hours=24),
            sample_step=datetime.timedelta(minutes=15),
            plot_title=global_start_time.strftime("GHI @ %Y-%m-%d"),
        )
        assert plot_data.shape[0] == archive_lut_size
    display_data = []
    for array_idx in tqdm.tqdm(range(archive_lut_size), desc="reshaping for final display"):
        display = cv.vconcat([raw_data[array_idx, ch_idx, ...] for ch_idx in range(len(channels))])
        while any([s > 1200 for s in display.shape]):
            display = cv.resize(display, (-1, -1), fx=0.75, fy=0.75)
        if plot_data is not None:
            plot = plot_data[array_idx]
            plot_scale = display.shape[0] / plot.shape[0]
            plot = cv.resize(plot, (-1, -1), fx=plot_scale, fy=plot_scale)
            display = cv.hconcat([display, plot])
        display_data.append(display)
    display = np.stack(display_data)
    array_idx, window_name, paused = 0, hdf5_path.split("/")[-1], False
    while True:
        cv.imshow(window_name, display[array_idx])
        ret = cv.waitKey(30 if not paused else 300)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == ord(' '):
            paused = ~paused
        if not paused or ret == ord('c'):
            array_idx = (array_idx + 1) % archive_lut_size


def preplot_live_ghi_curves(
        stations: typing.Dict[str, typing.Tuple],
        stations_data: typing.Dict[str, typing.Any],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        plot_title: typing.Optional[typing.AnyStr] = None,
) -> np.ndarray:
    """Pre-plots a set of GHI curves with update bars and returns the raw pixel arrays.
    This function is used in ``viz_hdf5_imagery`` to prepare GHI plots when stations & dataframe information
    is available.
    """
    plot_count = (window_end - window_start) // sample_step
    fig_size, fig_dpi, plot_row_count = (8, 6), 160, int(math.ceil(len(stations) / 2))
    plot_data = np.zeros((plot_count, fig_size[0] * fig_dpi, fig_size[1] * fig_dpi, 3), dtype=np.uint8)
    fig = plt.figure(num="ghi", figsize=fig_size[::-1], dpi=fig_dpi, facecolor="w", edgecolor="k")
    ax = fig.subplots(nrows=plot_row_count, ncols=2, sharex="all", sharey="all")
    art_handles, art_labels = [], []
    for station_idx, station_name in enumerate(stations):
        plot_row_idx, plot_col_idx = station_idx // 2, station_idx % 2
        ax[plot_row_idx, plot_col_idx] = plot_ghi_curves(
            clearsky_ghi=np.asarray(stations_data[station_name]["csky"]),
            station_ghi=np.asarray(stations_data[station_name]["ghi"]),
            pred_ghi=None,
            window_start=window_start,
            window_end=window_end - sample_step,
            sample_step=sample_step,
            horiz_offset=datetime.timedelta(hours=0),
            ax=ax[plot_row_idx, plot_col_idx],
            station_name=station_name,
            station_color=get_label_html_color_code(station_idx + 1),
            current_time=window_start
        )
        for handle, lbl in zip(*ax[plot_row_idx, plot_col_idx].get_legend_handles_labels()):
            # skipping over the duplicate labels messes up the legend, we must live with the warning
            art_labels.append("_" + lbl if lbl in art_labels or lbl == "current" else lbl)
            art_handles.append(handle)
    fig.autofmt_xdate()
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.legend(art_handles, labels=art_labels, loc="lower center", ncol=2)
    fig.canvas.draw()  # cache renderer with default call first
    subaxbgs = [fig.canvas.copy_from_bbox(subax.bbox) for subax in ax.flatten()]
    for idx in tqdm.tqdm(range(plot_count), desc="preparing ghi plots"):
        for subax, subaxbg in zip(ax.flatten(), subaxbgs):
            fig.canvas.restore_region(subaxbg)
            for handle, lbl in zip(*subax.get_legend_handles_labels()):
                if lbl == "current":
                    curr_time = matplotlib.dates.date2num(window_start + idx * sample_step)
                    handle.set_data([curr_time, curr_time], [0, 1])
                    subax.draw_artist(handle)
            fig.canvas.blit(subax.bbox)
        plot_data[idx, ...] = np.reshape(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8),
                                         (*(fig.canvas.get_width_height()[::-1]), 3))[..., ::-1]
    return plot_data


def plot_ghi_curves(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: typing.Optional[np.ndarray],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        horiz_offset: datetime.timedelta,
        ax: plt.Axes,
        station_name: typing.Optional[typing.AnyStr] = None,
        station_color: typing.Optional[typing.AnyStr] = None,
        current_time: typing.Optional[datetime.datetime] = None,
) -> plt.Axes:
    """Plots a set of GHI curves and returns the associated matplotlib axes object.
    This function is used in ``draw_daily_ghi`` and ``preplot_live_ghi_curves`` to create simple
    graphs of GHI curves (clearsky, measured, predicted).
    """
    assert clearsky_ghi.ndim == 1 and station_ghi.ndim == 1 and clearsky_ghi.size == station_ghi.size
    assert pred_ghi is None or (pred_ghi.ndim == 1 and clearsky_ghi.size == pred_ghi.size)
    hour_tick_locator = matplotlib.dates.HourLocator(interval=4)
    minute_tick_locator = matplotlib.dates.HourLocator(interval=1)
    datetime_fmt = matplotlib.dates.DateFormatter("%H:%M")
    datetime_range = pd.date_range(window_start, window_end, freq=sample_step)
    xrange_real = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if current_time is not None:
        ax.axvline(x=matplotlib.dates.date2num(current_time), color="r", label="current")
    station_name = f"measured ({station_name})" if station_name else "measured"
    ax.plot(xrange_real, clearsky_ghi, ":", label="clearsky")
    if station_color is not None:
        ax.plot(xrange_real, station_ghi, linestyle="solid", color=station_color, label=station_name)
    else:
        ax.plot(xrange_real, station_ghi, linestyle="solid", label=station_name)
    datetime_range = pd.date_range(window_start + horiz_offset, window_end + horiz_offset, freq=sample_step)
    xrange_offset = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if pred_ghi is not None:
        ax.plot(xrange_offset, pred_ghi, ".-", label="predicted")
    ax.xaxis.set_major_locator(hour_tick_locator)
    ax.xaxis.set_major_formatter(datetime_fmt)
    ax.xaxis.set_minor_locator(minute_tick_locator)
    hour_offset = datetime.timedelta(hours=1) // sample_step
    ax.set_xlim(xrange_real[hour_offset - 1], xrange_real[-hour_offset + 1])
    ax.format_xdata = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax.grid(True)
    return ax


def draw_daily_ghi(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: np.ndarray,
        stations: typing.Iterable[typing.AnyStr],
        horiz_deltas: typing.List[datetime.timedelta],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
):
    """Draws a set of 2D GHI curve plots and returns the associated matplotlib fig/axes objects.
    This function is used in ``viz_predictions`` to prepare the full-horizon, multi-station graphs of
    GHI values over numerous days.
    """
    assert clearsky_ghi.ndim == 2 and station_ghi.ndim == 2 and clearsky_ghi.shape == station_ghi.shape
    station_count = len(list(stations))
    sample_count = station_ghi.shape[1]
    assert clearsky_ghi.shape[0] == station_count and station_ghi.shape[0] == station_count
    assert pred_ghi.ndim == 3 and pred_ghi.shape[0] == station_count and pred_ghi.shape[2] == sample_count
    assert len(list(horiz_deltas)) == pred_ghi.shape[1]
    pred_horiz = pred_ghi.shape[1]
    fig = plt.figure(num="ghi", figsize=(18, 10), dpi=80, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.subplots(nrows=pred_horiz, ncols=station_count, sharex="all", sharey="all")
    handles, labels = None, None
    for horiz_idx in range(pred_horiz):
        for station_idx, station_name in enumerate(stations):
            ax[horiz_idx, station_idx] = plot_ghi_curves(
                clearsky_ghi=clearsky_ghi[station_idx],
                station_ghi=station_ghi[station_idx],
                pred_ghi=pred_ghi[station_idx, horiz_idx],
                window_start=window_start,
                window_end=window_end,
                sample_step=sample_step,
                horiz_offset=horiz_deltas[horiz_idx],
                ax=ax[horiz_idx, station_idx],
            )
            handles, labels = ax[horiz_idx, station_idx].get_legend_handles_labels()
    for station_idx, station_name in enumerate(stations):
        ax[0, station_idx].set_title(station_name)
    for horiz_idx, horiz_delta in zip(range(pred_horiz), horiz_deltas):
        ax[horiz_idx, 0].set_ylabel(f"GHI @ T+{horiz_delta}")
    window_center = window_start + (window_end - window_start) / 2
    fig.autofmt_xdate()
    fig.suptitle(window_center.strftime("%Y-%m-%d"), fontsize=14)
    fig.legend(handles, labels, loc="lower center")
    return fig2array(fig)


def viz_predictions(
        predictions_path: typing.AnyStr,
        dataframe_path: typing.AnyStr,
        test_config_path: typing.AnyStr,
):
    """Displays a looping visualization of the GHI predictions saved by the evaluation script.
    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The arrow keys allow the user to change which day is being shown.
    """
    assert os.path.isfile(test_config_path) and test_config_path.endswith(".json"), "invalid test config"
    with open(test_config_path, "r") as fd:
        test_config = json.load(fd)
    stations = test_config["stations"]
    target_datetimes = test_config["target_datetimes"]
    start_bound = datetime.datetime.fromisoformat(test_config["start_bound"])
    end_bound = datetime.datetime.fromisoformat(test_config["end_bound"])
    horiz_deltas = [pd.Timedelta(d).to_pytimedelta() for d in test_config["target_time_offsets"]]
    assert os.path.isfile(predictions_path), f"invalid preds file path: {predictions_path}"
    with open(predictions_path, "r") as fd:
        predictions = fd.readlines()
    assert len(predictions) == len(target_datetimes) * len(stations), \
        "predicted ghi sequence count mistmatch wrt target datetimes x station count"
    assert len(predictions) % len(stations) == 0
    predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    predictions = predictions.reshape((len(stations), len(target_datetimes), -1))
    pred_horiz = predictions.shape[-1]
    target_datetimes = pd.DatetimeIndex([datetime.datetime.fromisoformat(t) for t in target_datetimes])
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)
    dataframe = dataframe[dataframe.index >= start_bound]
    dataframe = dataframe[dataframe.index < end_bound]
    assert dataframe.index.get_loc(start_bound) == 0, "invalid start bound (should land at first index)"
    assert len(dataframe.index.intersection(target_datetimes)) == len(target_datetimes), \
        "bad dataframe target datetimes overlap, index values are missing"
    # we will display 24-hour slices with some overlap (configured via hard-coded param below)
    time_window, time_overlap, time_sample = \
        datetime.timedelta(hours=24), datetime.timedelta(hours=3), datetime.timedelta(minutes=15)
    assert len(dataframe.asfreq("15min").index) == len(dataframe.index), \
        "invalid dataframe index padding (should have an entry every 15 mins)"
    sample_count = ((time_window + 2 * time_overlap) // time_sample) + 1
    day_count = int(math.ceil((end_bound - start_bound) / time_window))
    clearsky_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    station_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    pred_ghi_data = np.full((day_count, len(stations), pred_horiz, sample_count), fill_value=float("nan"), dtype=np.float32)
    days_range = pd.date_range(start_bound, end_bound, freq=time_window, closed="left")
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing daytime GHI intervals")):
        window_start, window_end = day_start - time_overlap, day_start + time_window + time_overlap
        sample_start, sample_end = (window_start - start_bound) // time_sample, (window_end - start_bound) // time_sample
        for sample_iter_idx, sample_idx in enumerate(range(sample_start, sample_end + 1)):
            if sample_idx < 0 or sample_idx >= len(dataframe.index):
                continue
            sample_row = dataframe.iloc[sample_idx]
            sample_time = window_start + sample_iter_idx * time_sample
            target_iter_idx = target_datetimes.get_loc(sample_time) if sample_time in target_datetimes else None
            for station_idx, station_name in enumerate(stations):
                clearsky_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_CLEARSKY_GHI"]
                station_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_GHI"]
                if target_iter_idx is not None:
                    pred_ghi_data[day_idx, station_idx, :, sample_iter_idx] = predictions[station_idx, target_iter_idx]
    displays = []
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing plots")):
        displays.append(draw_daily_ghi(
            clearsky_ghi=clearsky_ghi_data[day_idx],
            station_ghi=station_ghi_data[day_idx],
            pred_ghi=pred_ghi_data[day_idx],
            stations=stations,
            horiz_deltas=horiz_deltas,
            window_start=(day_start - time_overlap),
            window_end=(day_start + time_window + time_overlap),
            sample_step=time_sample,
        ))
    display = np.stack(displays)
    day_idx = 0
    while True:
        cv.imshow("ghi", display[day_idx])
        ret = cv.waitKey(100)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == 81 or ret == 84:  # UNIX: left or down arrow
            day_idx = max(day_idx - 1, 0)
        elif ret == 82 or ret == 83:  # UNIX: right or up arrow
            day_idx = min(day_idx + 1, len(displays) - 1)
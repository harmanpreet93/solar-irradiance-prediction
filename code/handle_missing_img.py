import utils
import numpy as np
import pandas as pd
import cv2
import h5py
import json
import typing
from pathlib import Path
from scipy import interpolate

def interpolate_images(data):
    print("Interpolating missing image ...")
    data2 = np.copy(data.astype(float))
    for i in range(0, data.shape[1]):
        col_values = data2[:, i]
        col_idx = np.arange(len(col_values))
        # idx of non-zero values
        idx = np.where(col_values != 0)
        # create instance of interp1d
        interp = interpolate.interp1d(col_idx[idx], col_values[idx], kind='linear')
        # interpolate column wise
        new_col_values = interp(col_idx)
        data2[:, i] = new_col_values

    return data2

def handle_missing_img(hdf5_file, channel):
    # get LUT dataset
    ch_lut_name = channel + "_LUT"
    ch_lut = hdf5_file[ch_lut_name][()]
    # identify the offset of missing images
    missing_idx = np.where(np.equal(ch_lut, -1))

    # flatten and stack the images vertically
    flat_img = np.zeros((650*1500))
    for i, j in enumerate(ch_lut):
        # when there is a missing image, replace it by a zero vector
        if j == -1:
            flat_img = np.vstack((flat_img, np.zeros((650*1500))))
        else:
            fetched_data = np.ravel(utils.fetch_hdf5_sample(channel, hdf5_file, i))
            flat_img = np.vstack((flat_img, fetched_data))

    interp_img = interpolate_images(flat_img[1:])

    return interp_img

def handle_missing_img2(vec1, vec2):

    # flatten and stack the images vertically
    missing_vec = np.ravel(np.zeros(vec1.shape))
    stack = np.vstack((np.ravel(vec1), missing_vec, np.ravel(vec2)))
    interp_img = interpolate_images(stack)

    return interp_img[1]

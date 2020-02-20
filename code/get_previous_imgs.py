def get_previous_imgs(current_offset, h5_data, seq_len):
    # We assume the previous img are within the same day
    previous_imgs= np.zeros(shape=(seq_len, 650, 1500, 5))
    for i in range(current_offset - seq_len, current_offset):
        prev_img_ch1 = utils.fetch_hdf5_sample("ch1", h5_data, current_offset - i)
        prev_img_ch2 = utils.fetch_hdf5_sample("ch2", h5_data, current_offset - i)
        prev_img_ch3 = utils.fetch_hdf5_sample("ch3", h5_data, current_offset - i)
        prev_img_ch4 = utils.fetch_hdf5_sample("ch4", h5_data, current_offset - i)
        prev_img_ch6 = utils.fetch_hdf5_sample("ch6", h5_data, current_offset - i)
        previous_imgs[i] = np.stack((prev_img_ch1, prev_img_ch2, prev_img_ch3, \
        prev_img_ch4, prev_img_ch6), axis=-1)
    return previous_imgs
     
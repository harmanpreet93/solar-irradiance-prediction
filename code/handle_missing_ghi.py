import numpy as np
import pandas as pd


# We define the following method to to complete the missing GHI value
def fill_missing_ghi_linear(df, timestamp, station_id):
    ''' returns a dataframe with GHI values around the given timestamp.
        if missing GHI before or after this timestamp, it will be interpolated.
        it returns a sorted DF with timestamps indexes.
        Look for the missing one to get the interpolated GHI.
    '''
    ghi_col = station_id+"_GHI"
    cur_offset = df.loc[timestamp]["hdf5_8bit_offset"]
    GHI_values = pd.DataFrame(columns=[ghi_col])
    ts = timestamp
    GHI_values.at[ts] = df.loc[df.index == ts][ghi_col].values

    # Get previous and next available GHI value
    for i in range(1, np.int(cur_offset)):
        ts = ts - pd.DateOffset(minutes=15)
        prev_val = df.loc[df.index == ts][ghi_col].values
        if np.isnan(prev_val):
            GHI_values.at[ts] = np.NaN
        else:
            GHI_values.at[ts] = prev_val
            break
    ts = timestamp
    for j in range(1, np.int(95-cur_offset)):
        ts = ts + pd.DateOffset(minutes=15)
        next_val = df.loc[df.index == ts][ghi_col].values
        if np.isnan(next_val):
            GHI_values.at[ts] = np.NaN
        else:
            GHI_values.at[ts] = next_val
            break
    if len(GHI_values) == 0:
        print("No available values")
    else:
        # interpolate current GHI value
        GHI_values.sort_index(inplace=True)

    GHI_values = GHI_values[ghi_col].interpolate(method='linear', limit_direction='both')

    for i, j in enumerate(GHI_values.index):
        df.at[j, ghi_col] = GHI_values.iloc[i]

    return df


def fill_missing_ghi_mean(df, timestamp, station_id):
    ''' returns a dataframe with GHI values that are calculated as the mean
        of the day before and the day after for the same timestamp.
    '''
    ghi_col = station_id+"_GHI"

    cur_day = df.loc[df.index.date == timestamp.date()]

    ts_day_befor = timestamp - pd.DateOffset(day=1)
    ts_day_after = timestamp + pd.DateOffset(day=1)

    get_previous_day = df[df.index.date == ts_day_befor][ghi_col]
    get_next_day = df[df.index.date == ts_day_after][ghi_col]

    day_concat = pd.concat((get_previous_day, get_next_day))

    mean_ghi = day_concat.groupby(day_concat.index.time).mean()

    # Get previous and next available GHI value
    for i, j in enumerate(cur_day.index):
        if np.isnan(cur_day.iloc[i][ghi_col]):
            df.at[j, ghi_col] = mean_ghi.iloc[i]

    return df

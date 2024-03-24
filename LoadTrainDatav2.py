#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool
import time
from tqdm import tqdm

import sys

# year = sys.argv[1]
year = '2020'

# Read csv's, Replace NA and negative values
def readCSV(all_files):
    li = []
    for filename in tqdm(all_files):
        df = pd.read_csv(filename, index_col=None, header=0)
        df = df.iloc[: , 1:]
        # df.drop(['PID','PID.1',"cell",'weight','Date','X','Y','CROP'], axis=1, inplace=True)
        # df.drop(['id','PID','Date','FREQ','YEAR'], axis=1, inplace=True)
        # Select columns: band1, band2, band3
        df = df[['band1','band2','band3']]
        li.append(df)
    sample = pd.read_csv(filename)
    class_sample = sample['CROP']
    frame = pd.concat(li,axis=1)
    #frame = frame.fillna(0)
    frame = frame.astype(int)
    frame[frame == 0] = np.nan
    return frame,class_sample

# Interpolation function
def interpolate(pixel,bands=3):

    arr = np.array(pixel)
    arr = arr.reshape((seq_len,bands))
    arr_T = arr.T

    # Create a new array with the same shape as the original array
    dfs = []
    
    for i in range(bands):
        # Get the corresponding band
        df_band = pd.DataFrame(arr_T[i])
        #print(df_band)
        # Check if there is atleast one non-zero value
        if df_band.isna().sum()[0] < len(df_band) :
            # Replace 0 with nan
            df_band = df_band.replace(0,np.nan)

            # Interpolate the band
            df_band = df_band.interpolate(method='linear')
            # Replace the interpolated values nans using bfill
            df_band = df_band.replace(to_replace=np.nan, method='bfill')

        # Transpose the array
        df_band = df_band.T

        dfs.append(df_band)
    
    # Convert list to tuple and concatenate the bands
    cat = np.concatenate(tuple(dfs),axis=0)
    cat = cat.T
    cat = cat.reshape(seq_len*bands,)
    new_band = pd.DataFrame(cat)

    for i in range(len(pixel)):
        pixel[i] = new_band.iloc[i]
        
    return pixel


# Main function
def main():

    csv_path = "TrainData/TrainingCSVData/"+year
    all_files = sorted(glob.glob(csv_path + "/*.csv"))
    bands = 3

    group_numbers = []
    for file in all_files:
        group_number = (pd.to_datetime(file[-12:-4]).day_of_year)//7
        group_numbers.append(group_number)
    group_numbers = pd.DataFrame(group_numbers).T

    # Load the csvs into array
    print("Loading csv's...")
    frame,class_sample = readCSV(all_files)
    frame = frame.T.reset_index(drop=True).T
    print("Loaded csv's")

    band1_columns = frame.iloc[:,::3]
    band2_columns = frame.iloc[:,1::3]
    band3_columns = frame.iloc[:,2::3]

    band1_columns = band1_columns.T.reset_index(drop=True).T
    band2_columns = band2_columns.T.reset_index(drop=True).T
    band3_columns = band3_columns.T.reset_index(drop=True).T

    band1_concat = pd.concat([group_numbers,band1_columns],ignore_index=True).T
    band2_concat = pd.concat([group_numbers,band2_columns],ignore_index=True).T
    band3_concat = pd.concat([group_numbers,band3_columns],ignore_index=True).T

    band1_max = band1_concat.groupby(0).apply(lambda x: x.max()).T
    band2_max = band2_concat.groupby(0).apply(lambda x: x.max()).T
    band3_max = band3_concat.groupby(0).apply(lambda x: x.max()).T

    band1_max.columns = ['band1_'+str(i) for i in range(band1_max.shape[1])]
    band2_max.columns = ['band2_'+str(i) for i in range(band2_max.shape[1])]
    band3_max.columns = ['band3_'+str(i) for i in range(band3_max.shape[1])]
    
    band_concat_org = pd.concat([band1_max,band2_max,band3_max],axis=1)
    band_concat = band_concat_org[list(sum(zip(band1_max.columns, band2_max.columns, band3_max.columns), ()))]
    # Remove first row
    band_concat = band_concat.iloc[1:,:]

    print("Interpolating...")
    
    # Do the interpolation. Use multiple cores by using map elastic function
    t1 = time.time()
    result_arr = []

    with Pool() as p:
        result_arr = p.map(interpolate,[band_concat.iloc[i] for i in tqdm(range(len(band_concat)))])
    # result_arr = [interpolate(band_concat.iloc[i]) for i in tqdm(range(len(band_concat)))]


    print("Time Took for Interpolation in seconds:",time.time()-t1)
    
    
    # Reshape the array, Convert to dataframe and save the pickle
    result_arr = np.reshape(result_arr,(len(result_arr),seq_len*bands))
    df = pd.DataFrame(result_arr)
    new_frame = pd.concat([df,class_sample],axis=1)
    new_frame.to_pickle("TrainData/Train_no_abs2"+year+".pkl")
    print("Saved interpolated data")

    print('Completed')

seq_len = 366//7 + 1
if __name__ == "__main__":
    main()
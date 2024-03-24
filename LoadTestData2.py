import numpy as np
# import gdal
import rasterio
import scipy
import os
import glob
import pandas as pd
from numpy import expand_dims
from numpy import moveaxis
from multiprocessing import Pool
import time
from tqdm import tqdm
import sys

# Read year as input argument
# year = sys.argv[1]
year = '2017'

# Read all temporal images from dir using rasterio and append to list 
data_df = pd.DataFrame()
band1, band2, band3 = [], [], []
path = "final_img/"+str(year)+"/S1A*.tif"
files = sorted(glob.iglob(path))

print("Reading group numbers...")

group_dict = {}
for file in tqdm(files):
    day_of_the_year = pd.to_datetime(file[-12:-4]).day_of_year
    group_number = day_of_the_year//7
    if group_number not in group_dict:
        
        group_dict[group_number] = []
    group_dict[group_number].append(file)

def read_image(index):
    value = group_dict[index]
    band1, band2, band3 = [], [], []
    
    for file in value:
        with rasterio.open(file,'r') as src:
            band1.append(src.read(1))
            band2.append(src.read(2))
            band3.append(src.read(3))

    # Create a numpy array from the list of arrays
    band1 = np.dstack(band1)
    band2 = np.dstack(band2)
    band3 = np.dstack(band3)

    band1 = np.where(band1 == -32768,32767, band1)
    band2 = np.where(band2 == -32768,32767, band2)
    # band3 = np.where(band3 == -32768,32767, band3)

    band1 = np.abs(band1)
    band2 = np.abs(band2)
    # band3 = np.abs(band3)

    #Perform element-wise max operation on the each band
    band1 = np.max(band1, axis=2)
    band2 = np.max(band2, axis=2)
    band3 = np.max(band3, axis=2)

    return band1, band2, band3

print('Reading images...')
t = time.time()
with Pool(processes=len(group_dict)) as pool:
    result = pool.map(read_image, range(len(group_dict)))
print(time.time()-t)

# rasterio.errors.RasterioIOError: Read or write failed. final_img/2017/S1A_IW_GRDH_1SDV_20170919.tif, band 1: IReadBlock failed at X offset 0, Y offset 164: TIFFReadEncodedStrip() failed.
# If this error occurs, then there is an issue with parallel processing. Comment above line and uncomment below line to run sequentially.

# result = [read_image(i) for i in tqdm(range(len(group_dict)))]

print('Creating dataframe...')

result = np.array(result)

print('Result shape:', result.shape)
print('Saving to file...')
# Save the result to a numpy file
np.save('TestData/LoadedTestData'+year+'.npy', result)
print('Saved to file!')
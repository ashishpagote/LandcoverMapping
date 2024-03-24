import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool

import sys
import pickle as pkl
import os

# Read year as input argument
year = sys.argv[1]

data = np.load('TestData/LoadedTestData'+year+'.npy')

data = np.transpose(data, (2,3,1,0))

print('Interpolating...')

# Interpolation similar to training data
def interpolate(index,bands=3):    
    arr = data[index]
    bands, seq_len = arr.shape
    dfs = []
    
    for i in range(bands):
        df_band = pd.DataFrame(arr[i])
        if df_band.sum().sum() > 0:
            df_band = df_band.replace(0,np.nan)
            df_band = df_band.interpolate(method='linear')
            df_band = df_band.replace(to_replace=np.nan, method='bfill')
        df_band = df_band.T
        dfs.append(df_band)
        
    # Convert list to tuple and concatenate the bands
    cat = np.concatenate(tuple(dfs),axis=0)
    return cat

t1 = time.time()
with Pool(processes=70) as pool:
    result = pool.map(interpolate,[(i,j) for i in range(len(data)) for j in range(len(data[0]))])
t2 = time.time()
print("Took:",time.time()-t1)

with open('TestData/Result2017.pkl','wb') as f:
    pkl.dump(result,f)

result = np.reshape(result,(len(data),len(data[0]),3,53))
print('Result shape:', result.shape)

print('Saving to file...')
with open('TestData/Test'+year+'.npy', 'wb') as f:
    np.save(f,result)
print('Done!')

os.system('rm TestData/Result2017.pkl')

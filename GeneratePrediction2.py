import os
import pandas as pd
import numpy as np
import time
import torch
from torch.autograd import Variable
from tqdm import tqdm
from LSTM import LSTM1

import sys
import argparse

parser = argparse.ArgumentParser(description='Generate Predicitons LSTM')
parser.add_argument('--year', type=int, default=2018, help='Year to generate predictions')
parser.add_argument('--exp_name', type=str, default='new_all_years_crop_mask', help='Name of experiment')

opt = parser.parse_args()
year = opt.year
exp_name = opt.exp_name

print("Create Required Directories if they do not exist")
# Predctions directory
if not os.path.exists('./Predictions'):
    os.makedirs('./Predictions')

# Create Predictions/exp_name directory if it does not exist
if not os.path.exists('./Predictions/' + exp_name):
    os.makedirs('./Predictions/' + exp_name)




print('Loading data...')
# Load interpolated test data
test_ = np.load("TestData/Test" + str(year) + ".npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 53
learning_rate = 0.001 #0.001 lr

input_size = 3 #number of features/bands
hidden_size = 100 #number of features in hidden state
num_layers = 2 #number of stacked lstm layers

model_name = "best_model_" + exp_name + ".pth"
model_path = os.path.join("./best_models", model_name)


years = [2018, 2019, 2020]
dataset = pd.DataFrame()
for y in years:
    dataset_year = pd.read_pickle("TrainData/Train_no_abs2" + str(y) + ".pkl")
    dataset = pd.concat([dataset, dataset_year])

columns_to_check = dataset.columns[dataset.columns != 'CROP']
print(len(dataset))

print('Drop rows with all NA')
dataset = dataset.dropna(subset=columns_to_check, how='all')

num_classes = dataset['CROP'].nunique() 
print(num_classes)

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers,seq_len)
lstm1.to(device)#our lstm class 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) #?? To be commented out

print('Loading model...')
# Load trained model state weights
if device == "cuda":
    lstm1.load_state_dict(torch.load(model_path)['model_state_dict'])
else:
    lstm1.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

lstm1.to(device)
print('Evaulation started...')
lstm1.eval()

print('Predicting...')
# Generate predictions
t1 = time.time()
# Create an empty array of image size
predictions = np.zeros((test_.shape[0],test_.shape[1]),dtype=np.float32)
# Looping though every row
for i in tqdm(range(len(test_))):
    X_test_tensors = Variable(torch.Tensor(test_[i]))
    X_test_tensors_final = X_test_tensors.transpose(1,2).to(device)
    with torch.no_grad():
        outputs = lstm1(X_test_tensors_final)
        _,predicted = torch.max(outputs.data,1)
        predicted_numpy = predicted.cpu().numpy().squeeze()
        predictions[i,:] = predicted_numpy
predictions = predictions.astype('int16')
print("Took:",time.time()-t1)

print('Saving predictions...')
save_path = os.path.join('./Predictions/', exp_name, 'Prediction'+str(year)+'.npy')
np.save(save_path, predictions)

print("Done")


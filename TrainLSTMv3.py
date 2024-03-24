import pandas as pd

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from LSTM import LSTM1
import pickle
# import argparse
torch.cuda.set_device("cuda:1")

from tqdm import tqdm

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
#print(config['epochs'])

# parser = argparse.ArgumentParser(description='Train LSTM')
# parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
# parser.add_argument('--epochs', type=int, default=500, help='Total number of epochs to train')
# parser.add_argument('--batch_size', type=int, default=16384, help='Batch size for training')
# parser.add_argument('--exp_name', type=str, default='default', help='Name of experiment')
# parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in LSTM')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
# opt = parser.parse_args()

# EPOCHS = opt.epochs
# BATCH_SIZE = opt.batch_size
# exp_name = opt.exp_name
# num_layers = opt.num_layers
# learning_rate = opt.learning_rate
# resume = opt.resume

EPOCHS = int(config['TRAIN']['epochs'])
BATCH_SIZE = int(config['TRAIN']['batch_size'])
exp_name = config['TRAIN']['exp_name']
num_layers = int(config['TRAIN']['num_layers'])
learning_rate = float(config['TRAIN']['learning_rate'])
resume = True if config['TRAIN']['resume'] == 'True' else False
hidden_size = int(config['TRAIN']['hidden_size'])
input_dim = int(config['TRAIN']['input_dim'])

print('Create required directories if they do not exist')
# Saved model directory
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')
    
# Create saved_models/exp_name directory if it does not exist
if not os.path.exists('./saved_models/' + exp_name):
    os.makedirs('./saved_models/' + exp_name)

# Create best_model directory if it does not exist
if not os.path.exists('./best_models'):
    os.makedirs('./best_models')

# Create result_plots directory if it does not exist
if not os.path.exists('./result_plots'):
    os.makedirs('./result_plots')

# Create result_plots/exp_name directory if it does not exist
if not os.path.exists('./result_plots/' + exp_name):
    os.makedirs('./result_plots/' + exp_name)

print('Importing data...')


# Create empty dataframe to store results
dataset = pd.DataFrame()
years = ['2018','2019','2020']
for year in years:
    dataset_year = pd.read_pickle("TrainData/Train_no_abs2"+year+".pkl")
    #dataset_year = pd.read_pickle("TrainData/Train"+year+".pkl")
    dataset = pd.concat([dataset, dataset_year])

# dataset.drop(dataset.columns[range(2, len(dataset.columns), 3)], axis=1 ,inplace=True)

print('Concatenated data')

#print('Labels: ', dataset['CROP'].unique())
le = LabelEncoder()
#dataset=dataset[dataset['CROP'].isin(['Rice','Sugarcane'])]
#print('Labels: ', dataset['CROP'].unique())
columns_to_check = dataset.columns[dataset.columns != 'CROP']
print(len(dataset))

print('Drop rows with all NA')
dataset = dataset.dropna(subset=columns_to_check, how='all')

dataset['CROP'] = le.fit_transform(dataset['CROP'])
print(len(dataset))
print('Labels encoded: ', dataset['CROP'].unique())

#scaling min-max
dataset_1=dataset.loc[:, dataset.columns != 'CROP']
dataset_2=dataset[['CROP']]

dataset_1=(dataset_1-dataset_1.min())/(dataset_1.max()-dataset_1.min())
dataset=pd.concat([dataset_1,dataset_2],axis=1)
dataset=dataset.fillna(0)
print(dataset.describe())


train,test = train_test_split(dataset, test_size=0.1,shuffle=True,stratify=dataset['CROP'])
train, val = train_test_split(train, test_size=0.25,shuffle=True,stratify=train['CROP'])

print('Training set size: ', train.shape)
print('Validation set size: ', val.shape)
print('Test set size: ', test.shape)

seq_len = 366//7 +1
bands = input_dim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_classes = len(dataset['CROP'].unique()) #number of output classes 


x_train, y_train = train.iloc[:,:-1].to_numpy(), train.iloc[:,-1].to_numpy()
X_Train_tensors, Y_Train_tensors = Variable(torch.Tensor(x_train)), Variable(torch.Tensor(y_train))
X_train_tensors_final = torch.reshape(X_Train_tensors, (X_Train_tensors.shape[0],seq_len,bands)).to(device)
Y_train_tensors_final = Y_Train_tensors.to(device)
train_dataset = TensorDataset(X_train_tensors_final,Y_train_tensors_final) # create your datset
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE) # create your dataloader

x_val, y_val = val.iloc[:,:-1].to_numpy(), val.iloc[:,-1].to_numpy()
X_Val_tensors, Y_Val_tensors = Variable(torch.Tensor(x_val)), Variable(torch.Tensor(y_val))
X_val_tensors_final = torch.reshape(X_Val_tensors, (X_Val_tensors.shape[0],seq_len,bands)).to(device)
Y_val_tensors_final = Y_Val_tensors.to(device)
val_dataset = TensorDataset(X_val_tensors_final,Y_val_tensors_final) # create your datset
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE) # create your dataloader

x_test, y_test = test.iloc[:,:-1].to_numpy(), test.iloc[:,-1].to_numpy()
X_Test_tensors, Y_Test_tensors = Variable(torch.Tensor(x_test)), Variable(torch.Tensor(y_test))
X_test_tensors_final = torch.reshape(X_Test_tensors, (X_Test_tensors.shape[0],seq_len,bands)).to(device)
Y_test_tensors_final = Y_Test_tensors.to(device)
test_dataset = TensorDataset(X_test_tensors_final,Y_test_tensors_final) # create your datset
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE) # create your dataloader


# Import class_weight to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
    # 'balanced', np.unique(train['CROP']), train['CROP'])

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print('Class weights: ', class_weights)

lstm1 = LSTM1(
    num_classes, 
    input_dim, 
    hidden_size, 
    num_layers, 
    X_train_tensors_final.shape[1]
)
lstm1.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

START_EPOCH = 0
best_val_loss = np.inf

if resume:
    saved_models = glob(os.path.join('./saved_models', exp_name, '*.pth'))
    completed_epochs = [int(model_name.split('/')[-1].split('_')[-1].split('.pth')[0]) for model_name in saved_models]
    last_epoch = sorted(completed_epochs)[-1]

    last_model = torch.load(os.path.join('./saved_models', exp_name, 'model_epoch_' + str(last_epoch) + '.pth'),map_location=device)

    lstm1.load_state_dict(last_model['model_state_dict'])
    optimizer.load_state_dict(last_model['optimizer_state_dict'])

    START_EPOCH = last_model['epoch']+1
    best_val_loss = last_model['validation_loss']



print('Training starting')
# Train the model
train_losses = []
val_losses = []



for epoch in tqdm(range(START_EPOCH, EPOCHS)):
    batch_losses = []
    train_acc = 0.0
    lstm1.train()
    for inputs,labels in train_loader:
        inputs,labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        outputs = lstm1(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_train_loss = np.mean(batch_losses)
    train_losses.append(epoch_train_loss)
    lstm1.eval()
    with torch.no_grad():
        batch_val_losses = []
        for x_val,y_val in val_loader:
            x_val,y_val = x_val.to(device), y_val.to(device)
            y_val = y_val.long() 
            y_pred = lstm1(x_val)
            val_loss = criterion(y_pred,y_val)
            batch_val_losses.append(val_loss.item())
        validation_loss = np.mean(batch_val_losses)
        val_losses.append(validation_loss)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': lstm1.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'validation_loss': best_val_loss
                }, 
                os.path.join('./best_models','best_model_' + exp_name + '.pth')
            )

    # Save the model every 50 epochs
    if epoch % 10 == 0:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': lstm1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': validation_loss
            }, 
            os.path.join('./saved_models', exp_name, 'model_epoch_' + str(epoch) + '.pth')
        )


print('Training complete')
print('Training loss: ', epoch_train_loss)
print('Validation loss: ', validation_loss)
print('Saving Final model')
torch.save(
    {
        'epoch': epoch,
        'model_state_dict': lstm1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': validation_loss
    },
    os.path.join('./saved_models', exp_name, 'model_epoch_' + str(epoch) + '.pth')
)
print('Model saved')

print('Plotting Training loss')

# Clean up the plot
plt.clf()
# Plot and save the training loss
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig(os.path.join('./result_plots', exp_name, 'loss.png'))

print('Training Loss plot saved')

print('Evaluating Test set')
lstm1.eval()
predlist=torch.zeros(0,dtype=torch.long)
lbllist=torch.zeros(0,dtype=torch.long)
# nb_classes = 9
confusion_mat = torch.zeros(num_classes,num_classes)
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels = inputs.to(device), labels.to(device)
        outputs = lstm1(inputs)
        _,predicted = torch.max(outputs.data,1)
        predlist = torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist = torch.cat([lbllist,labels.view(-1).cpu()])
conf_mat = confusion_matrix(lbllist.numpy(),predlist.numpy())   

print(conf_mat)
# Save the confusion matrix as a pickle file
with open(os.path.join('./result_plots', exp_name, 'confusion_matrix.pkl'), 'wb') as f:
    pickle.dump(conf_mat, f)

# Clean up the plot
plt.clf()

sns.heatmap(conf_mat, annot=True)
plt.savefig(os.path.join('./result_plots', exp_name, 'confusion_matrix.png'))

# Generate accuracy for each class
class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)

# Save the class accuracy as a pickle file
with open(os.path.join('./result_plots', exp_name, 'class_accuracy.pkl'), 'wb') as f:
    pickle.dump(class_accuracy, f)
    

# Generate user accuracy and producer accuracy
user_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(0)
producer_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)

# Plot class accuracy, user accuracy and producer accuracy in a table
plt.clf()
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')

class_names =  ['Agroforestry', 'Cassava', 'Forest', 'Maize', 'Orchard', 'Other crop', 'Rice', 'Shrubland', 'Sugarcane', 'Urban', 'Water', 'Wetland']
the_table = ax.table(cellText=[class_accuracy, user_accuracy, producer_accuracy], rowLabels=['Class Accuracy', 'User Accuracy', 'Producer Accuracy'], colLabels=class_names, loc='center')
plt.savefig(os.path.join('./result_plots', exp_name, 'accuracy_table.png'))

labels = le.inverse_transform(np.arange(num_classes))


# Clear plot
plt.clf()

plt.bar(range(num_classes), class_accuracy)
plt.title('Class Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy %')
plt.xticks(range(num_classes), labels, rotation='vertical')
# Add rounded values inside the bar
for i in range(num_classes):
    plt.text(i, class_accuracy[i], str(round(class_accuracy[i], 2)), color='black', ha='center', va='bottom')

# Increase the space between the bars
plt.subplots_adjust(bottom=0.01)

plt.savefig(os.path.join('./result_plots', exp_name, 'class_accuracy.png'))
print('Class Accuracy plot saved')

print('Done!')


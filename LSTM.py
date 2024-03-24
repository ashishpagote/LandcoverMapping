import torch.nn as nn
from torch.autograd import Variable
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM network
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.batchNorm = nn.BatchNorm1d(seq_length)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1

        self.relu = nn.ReLU()
        # self.fc = nn.Linear(128,num_classes) #fully connected 2

        self.fc = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).requires_grad_().to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).requires_grad_().to(device) #internal state

        # # Select only the first input_size dimensions
        # x = x[:, :, :self.input_size]
        # Propagate input through LSTM
        #print(x.shape)
        norm = self.batchNorm(x)
        #print(norm)
        #print(1)
        output, _ = self.lstm(norm, (h_0, c_0)) #lstm with input, hidden, and internal state
        #print('output shape')
        #print(output.shape)

        # fc1 = self.fc_1(output[:, -1, :]) #fully connected 1
        # relu = self.relu(fc1) #relu
        # out = self.fc(relu) #fully connected 2

        out = self.fc(output[:,-1,:])
        return out
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, batch_size):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes # number of classes
        self.num_layers = num_layers # number of layers
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # hidden state
        self.seq_length = seq_length # sequence length
        self.batch_size = batch_size # batch size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm

        self.pipeline = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(hidden_size, 128), # fully connected 1
            nn.ReLU(),
            # nn.Linear(128, 64), # fully connected 2
            # nn.ReLU(),
            # nn.Linear(64, 32), # fully connected 3
            # nn.ReLU(),
            nn.Linear(128, num_classes) # fully connected last layer
        )
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        output = output.contiguous().view(x.size(0) * self.seq_length, self.hidden_size)
        return self.pipeline(output)
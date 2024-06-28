# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:36:44 2024
Author: Mason Lovejoy-Johnson
"""

"""
Generator:
    
    Used in the neural network to generate a sequence of data that
    will be sent to the discriminator to see if it is real or fake.
    This generator will use LSTM's to generate data.
    
"""

import torch
from torch import nn

class Generator(nn.Module):

    def __init__(self, feature_size, hidden_dim, window_size, target_size):
        super(Generator, self).__init__()
        self.q = target_size
        self.p = window_size
        self.feats = feature_size
        self.hid_dim = hidden_dim
        
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features = self.feats, out_features = self.hid_dim),
            nn.ReLU())
        self.lstm_1 = nn.LSTM(input_size = self.hid_dim, hidden_size = self.hid_dim, batch_first=True)
        
        # Curious if I can expand the input size in LSTM 2 for more information
        # in the neural network.
        # If I'm not noticing great results may want to change hidden_size???
        self.lstm_2 = nn.LSTM(input_size = self.feats, hidden_size = self.hid_dim, batch_first=True)
        self.ffn2 = nn.Sequential(
            nn.Linear(in_features = self.hid_dim, out_features = self.feats))
        
    def forward(self, x):
        # input_dim = (*, feature_size) * is any dimensions
        x = self.ffn(x)
        
        # input_dim = (window_size, feature_size)
        x, (hidden_state, cell_state) = self.lstm_1(x)
        
        # Do not need to keep cell_state and hidden_state but since it is already
        # in memory I'm fine with updating it in case I need to use it in the future
        
        noise = torch.randn(x.size(0), self.q, self.feats)
        # Also the inputs for lstm_2 should be noise so the lstm can 
        # generate a random sequence that the discriminator can use. At the moment.

        noise, (hidden_state, cell_state) = self.lstm_2(noise, (hidden_state, cell_state))
        output = self.ffn2(noise)
        return output
    
    def sample(self, x):
        return self.forward(x)




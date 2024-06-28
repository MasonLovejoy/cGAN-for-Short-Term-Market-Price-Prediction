# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:36:44 2024
Author: Mason Lovejoy-Johnson
"""

"""
Discriminator:
    
    Compares real data to fake data and decides which is which.
    hsummary
"""

import torch
import torch.nn as nn


"""
Notes for Model Optimization:
    
    (1) Play with filter numbers to see how that effects performance
    (2) In Conv2d make kernel_size = (3,1) to better find short term
        trends, but this will require me to change the input_size
    (3) Flatten and Linear may be played around with in order to 
        to increase information available to the model in the sigmoid
        function
"""

class Discriminator(nn.Module):
    
    " Discriminator Pytorch Module "
    
    # Channels actually equal 2 because I should create one for the
    # input and one for the generated noise
    # In future adjust kernel size
    
    def __init__(self, features, filter_num):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, filter_num, kernel_size=(1, features)),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(filter_num, filter_num, kernel_size=(3, 1)),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(filter_num, filter_num, kernel_size=(1,1)),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,1)),
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Dropout(p=0.1))
        self.layer5 = nn.Sequential(
            nn.Linear(filter_num, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        # the input should be a concatonated list with the inputs and generated
        # xshape = (Channels in, Height(window_size), Width(Feature Size))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.layer5(x)
        return output





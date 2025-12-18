# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:50:14 2024
Author: Mason Lovejoy-Johnson
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as py

from models import GeneratorPlanner
from models import DiscriminatorPlanner
from cGAN_training import CGANTrainer

path = 'data//nvda_data_tensor' 
data = torch.load(path).float()

def run_summary(loss_list: list, avg_loss_list: list, title: str) -> None:
    avg_min_loss = min(avg_loss_list)
    avg_min_idx = avg_loss_list.index(avg_min_loss)
    
    min_loss = min(loss_list)
    min_idx = loss_list.index(min_loss)
    
    x_axis = np.linspace(0, len(loss_list)-1, len(loss_list))
    y_axis = np.array(loss_list)
    
    avg_x_axis = np.linspace(0, len(loss_list)-1, len(avg_loss_list))
    avg_y_axis = np.array(avg_loss_list)
    
    print('__________________________________________________________\n')
    print(str(title)+' Run Summary: ')
    print('------------------------\n')
    print('Minimum Loss: '+str(min_loss)) 
    print('Iteration of Minimum Loss: '+str(min_idx))
    print('Average Run Summary: ')
    print('Minimum Average of 50 Run Loss: '+str(avg_min_loss))
    print('Iteration of Minimum Average of 50 Run Loss: '+str((avg_min_idx*50)+50))
    
    fig, (ax1, ax2) = py.subplots(2)
    
    ax1.plot(x_axis, y_axis, c='blue')
    ax2.plot(avg_x_axis, avg_y_axis, c='red')
    
    ax2.set_xlabel('Training Iterations')
     
    ax1.set_ylabel(str(title)+' Loss')
    ax2.set_ylabel('Average '+str(title)+' Loss')
     
    fig.suptitle(str(title)+' Loss Curve')
    py.show()
    
# can complete one full pass of the generator function,
# need to connect the discriminator
def sample_indices(dataset_size, batch_size):
    indices = np.random.choice(dataset_size, size=batch_size, replace=False)
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices

class cGAN(nn.Module):
    def __init__(self, feature_size, hidden_dim, filter_num, data, p, q, batch_size):
        super().__init__()
        self.x_real = data
        self.D_steps_per_G_step = 3
        self.G = Generator(feature_size, hidden_dim, p, q)
        self.D = Discriminator(feature_size, filter_num)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=8e-6, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-7, betas=(0, 0.9))
        self.p = p
        self.q = q
        self.batch_size = batch_size

        self.trainer = CGANTrainer(  # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer
            , p=self.p, q=self.q, 
        )
    
    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_real[indices, :self.p].clone()
            with torch.no_grad():
                x_fake = self.G.sample(x_past.clone())
                x_fake = torch.cat([x_past, x_fake], dim=1)
            D_loss_real, D_loss_fake, reg = self.trainer.D_trainstep(x_fake, self.x_real[indices])
            #if i == 0:
               # self.training_loss['D_loss_fake'].append(D_loss_fake)
               # self.training_loss['D_loss_real'].append(D_loss_real)

        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone()
        x_fake = self.G.sample(x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss = self.trainer.G_trainstep(x_fake_past, self.x_real[indices].clone())
        #self.training_loss['D_loss'].append(D_loss_fake + D_loss_real)
        #self.training_loss['G_loss'].append(G_loss)
        #self.evaluate(x_fake)
        return G_loss, D_loss_real, D_loss_fake
        
training_ig = cGAN(feature_size=42, hidden_dim=256, filter_num=4, data=data, p=3, q=1, batch_size=1)


g_loss_list = []
d_loss_list = []

avg_g_loss_list = []
avg_d_loss_list = []

temp_g_loss = []
temp_d_loss = []

for i in range(0, 40000):
    g_loss, d_loss_real, d_loss_fake = training_ig.step()
    
    if i % 50 == 0 and i != 0:
        avg_g_loss = np.average(temp_g_loss)
        avg_g_loss_list.append(avg_g_loss)
    
        avg_d_loss = np.average(temp_d_loss)
        avg_d_loss_list.append(avg_d_loss)
        
        print('__________________________________________________________\n')
        print('Iteration Number: '+str(i)+'\n')
        print('------------------------\n')
        print('Average Generator Loss: '+str(avg_g_loss)) 
        print('Average Discrimination Loss: '+str(avg_d_loss))
        
        temp_g_loss = []
        temp_d_loss = []
        
    temp_g_loss.append(g_loss)
    temp_d_loss.append(d_loss_real + d_loss_fake)
    
    g_loss_list.append(g_loss)
    d_loss_list.append(d_loss_real + d_loss_fake)
    
run_summary(g_loss_list, avg_g_loss_list, 'Generator')

run_summary(d_loss_list, avg_d_loss_list, 'Discriminator')

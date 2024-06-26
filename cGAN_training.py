# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:09:17 2024
Author: Mason Lovejoy-Johnson
"""
import functools
import torch
from torch import autograd

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
        
class CGANTrainer(object):
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
    ):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.p = p
        self.q = q

    def G_trainstep(self, x_fake, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        gloss = self.compute_loss(d_fake, 1)
        gloss = gloss + torch.mean((x_fake - x_real) ** 2)
        gloss.backward()
        self.G_optimizer.step()
        return gloss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Compute regularizer on fake/real
        dloss = dloss_fake + dloss_real
        dloss.backward()

        reg = torch.ones(1)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)
        return dloss_real.item(), dloss_fake.item(), reg.item()
    
    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)


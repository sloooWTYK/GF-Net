#!/usr/bin/env python
"""
model.py
--------
Physics Informed Neural Network for solving Poisson equation
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from problem import Problem
from options import Options

class Net(nn.Module):
    """
    Basic Network for PINNs
    """
    def __init__(self, layers, scale=1.0):
        """
        Initialization for Net
        """
        super().__init__()
        self.scale   = scale
        self.layers  = layers
        self.fcs     = []
        self.params  = []

        self.fc0 = nn.Linear(self.layers[0], self.layers[1], bias=False)
        setattr(self, f'fc{0}', self.fc0)

        weight = torch.tensor([
            [1.0, 0.0,  0.0,  0.0],
            [0.0, 1.0,  0.0,  0.0],
            [0.0, 0.0,  1.0,  0.0],
            [0.0, 0.0,  0.0,  1.0],
            [1.0, 0.0, -1.0,  0.0],
            [0.0, 1.0,  0.0, -1.0],
        ])
        
        self.fc0.weight = torch.nn.Parameter(weight)
        self.fc0.weight.requires_grad = False

        for i in range(1, len(layers) - 2):
            fc    = nn.Linear(self.layers[i], self.layers[i+1])
            setattr(self, f'fc{i}', fc)
            self._init_weights(fc)
            self.fcs.append(fc)
            
            param = nn.Parameter(torch.randn(self.layers[i+1]))
            setattr(self, f'param{i}', param)
            self.params.append(param)

        fc = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, f'fc{len(layers)-2}', fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    def _init_weights(self, layer):
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, X):
        X = self.fc0(X)
        for i in range(len(self.fcs)-1):
            X = self.fcs[i](X)
            X = torch.mul(self.params[i], X) * self.scale
            X = torch.sin(X)
        return self.fcs[-1](X)
    
class Net_PDE(nn.Module):
    def __init__(self, net, problem, device):
        super().__init__()
        self.net = net
        self.problem = problem
        self.device = device
        
    def forward(self, X):
        """
        Parameters:
        -----------
        X: torch tensor with shape [:, 4]
            interior samples
        """
        X.requires_grad_(True)
        u = self.net(X)

        X_symmetry = torch.cat((X[:, 2:], X[:, :2]), 1)
        u_symmetry = self.net(X_symmetry)
        res_symmetry = u - u_symmetry

        u_x  = torch.autograd.grad(u  , X, torch.ones_like(u), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u), create_graph=True)[0][:, [0]]

        u_y  = torch.autograd.grad(u  , X, torch.ones_like(u), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u), create_graph=True)[0][:, [1]]
        
        X.detach_()

        X_     = X.cpu().numpy()
        f      = self.problem.f(X_)
        a      = self.problem.a(X_)
        a_grad = self.problem.a(X_, order=1)
        a_x    = a_grad[:, [0]]
        a_y    = a_grad[:, [1]]
        r      = self.problem.r(X_)

        a      = torch.from_numpy(a).float().to(self.device)
        a_x    = torch.from_numpy(a_x).float().to(self.device)
        a_y    = torch.from_numpy(a_y).float().to(self.device)
        r      = torch.from_numpy(r).float().to(self.device)        
        f      = torch.from_numpy(f).float().to(self.device)
        
        res    = -(a * (u_xx + u_yy) + (a_x * u_x + a_y * u_y)) + r * u - f
        
        return res, res_symmetry

class Net_GRAD(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, X):
        """
        Parameters:
        -----------
        X: torch tensor with shape [:, 4]
            interior samples
        """
        X.requires_grad_(True)
        u = self.net(X)

        grad_u  = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, :2]
        
        X.detach_()
        
        return grad_u
    
    
class PINN(nn.Module):
    def __init__(self, net, problem, device):
        super().__init__()
        self.net = net        
        self.net_pde = Net_PDE(net, problem, device)

    def forward(self, X_interior, X_boundary):                
        res, res_symmetry = self.net_pde(X_interior)
        u_boundary        = self.net(X_boundary)

        return res, res_symmetry, u_boundary

    
if __name__ == '__main__':
    problem = Problem()
    layers = [4, 6, 50, 50, 1]
    args = Options().parse()
    net     = Net(layers)
    net_pde = Net_PDE(net, problem, device=args.device)
    pinn    = PINN(net, problem, device=args.device)
    params  = list(net.parameters())
    for name, value in net.named_parameters():
        print(name)
    # print(net.fc1.weight.shape)
    # print(net.fc1.bias)
        
    

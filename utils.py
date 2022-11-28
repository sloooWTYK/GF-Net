#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import shutil
import os

def in_subdomain(z_interior, subdomain):
    """ Return samples of x whose last two elements xi in subdomain
    
    Params
    ------
    z_interior (dict): including 'index' with shape (N, 1) and 'coord' with shape (N, 2)
    subdomain (tuple): (x0_min, x0_max, x1_min, x1_max) to determine the range of the subdomain
    
    Returns
    -------
    samples of x in subdomain
    """
    mask =  np.all([z_interior['coord'][:, 0] >= subdomain[0],
                    z_interior['coord'][:, 0] <= subdomain[1],
                    z_interior['coord'][:, 1] >= subdomain[2],
                    z_interior['coord'][:, 1] <= subdomain[3]], axis=0)
    return {'idx': z_interior['idx'][mask], 'coord': z_interior['coord'][mask]}

def calc_area(vertices):
    a = vertices[1] - vertices[0]
    b = vertices[2] - vertices[0]
    c = np.cross(a, b)
    return 0.5 * np.linalg.norm(c)

def tile(x, y):
    X = np.tile(x, (y.shape[0], 1))
    Y = np.vstack([np.tile(y[i], (x.shape[0], 1)) for i in range(y.shape[0])])
    
    return np.hstack((X, Y))

def show_image(x, u, elems):
    """
    show image using X and u

    params:
    ======
    x: torch tensor with device cuda and shape (:, 4)
       it should be transfered to cpu and then numpy ndarray
    u: torch tensor with device cuda and shape (:, 1)
       it should be firstly detached and then transfered to cpu and finally to numpy ndarray
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    tri = Triangulation(x[:, 0], x[:, 1], elems)
    surf = ax.plot_trisurf(tri, u[:, 0], cmap=plt.cm.Spectral)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$G$')
    # Add a color bar which map values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig

def save_checkpoints(k, state, is_best=None, save_dir=None):
    checkpoint = os.path.join(save_dir, f'checkpoint_{k}.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, f'{k}.pth.tar')
        shutil.copyfile(checkpoint, best_model) 
        
if __name__ == '__main__':
    x = np.random.rand(5, 2)
    y = np.random.rand(3, 2)
    z = tile(x, y)
    print(z)

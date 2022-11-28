#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from matplotlib.tri import Triangulation
import time
import os
import argparse
import math
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from problem import Poisson, DiffusionReaction
from dataset import Trainset, Validset
from model import Net, Net_PDE, PINN
from options import Options
from utils import show_image, save_checkpoints

class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.device       = args.device
        self.cuda_index   = args.cuda_index
        self.pde_case     = args.pde_case
        self.domain       = args.domain
        self.domain_type  = args.domain_type
        self.problem      = args.problem
        self.tol          = args.tol
        self.tol_change   = args.tol_change

        self.resume       = args.resume
        self.sigma        = args.sigma
        self.sigma_resume = args.sigma_resume
        self.scale        = args.scale
        self.scale_resume = args.scale_resume
        self.blocks_num   = args.blocks_num
        
        self.z_coarse_num = args.z_coarse_num
        self.x_coarse_num = args.x_coarse_num
        self.x_middle_num = args.x_middle_num
        self.x_refine_num = args.x_refine_num  

        # Trainset
        self.trainset     = Trainset(args)
        self.blocks       = self.trainset()
        self.blocks_info  = self.trainset._blocks_info()        
        print(self.trainset)
        
        # Criterion
        self.criterion    = nn.MSELoss()

        # HyperParameters setting
        self.epochs_Adam  = self.args.epochs_Adam
        self.epochs_LBFGS = self.args.epochs_LBFGS
        self.lam          = self.args.lam
        self.nn_name      = f'GFNet_{self.problem.sigma}'
        self.model_path   = self._model_path()
        
        # Learning rate
        self.lr           = self.args.lr
             

    def _model_path(self):
        """ Create directory of saved model"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        pde = os.path.join('checkpoints', f'{self.pde_case}_{self.domain_type}')
        if not os.path.exists(pde):
            os.mkdir(pde)
            
        network = os.path.join(pde, f'{self.nn_name}')
        if not os.path.exists(network):
            os.mkdir(network)

        z_sample = os.path.join(network, f'{self.z_coarse_num}')
        if not os.path.exists(z_sample):
            os.mkdir(z_sample)

        x_sample = os.path.join(z_sample, f'{self.x_coarse_num}_{self.x_middle_num}_{self.x_refine_num}')
        if not os.path.exists(x_sample):
            os.mkdir(x_sample)

        scale = os.path.join(x_sample, f'{self.scale[0]}_{self.scale[1]}')
        if not os.path.exists(scale):
            os.mkdir(scale)

        model_path = os.path.join(scale, f'{self.blocks_num[0]}_{self.blocks_num[1]}')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            
        return model_path        
        
    def train(self):
        """
        train process
        """        
        for k in range(len(self.blocks)):
            if self.blocks[k] is not None:
                self.train_block(k)
            else:
                print(f'Block #{k} is empty!!!')


    def train_block(self, k):

        # Networks
        self.net     = Net(args.layers)
        self.net_pde = Net_PDE(self.net, self.problem, self.device)
        self.pinn    = PINN(self.net, self.problem, self.device)
        
        # resume checkpoint if needed

        if self.resume:
            resume_path = os.path.join('checkpoints',
                                       f'{self.pde_case}_{self.domain_type}',
                                       f'GFNet_{self.sigma_resume}',
                                       f'{self.z_coarse_num}',
                                       f'{self.x_coarse_num}_{self.x_middle_num}_{self.x_refine_num}',
                                       f'{self.scale_resume[0]}_{self.scale_resume[1]}',
                                       f'{self.blocks_num[0]}_{self.blocks_num[1]}',
                                       f'{k}.pth.tar')
            if os.path.isfile(resume_path):
                print(f'Resuming training, loading {resume_path} ...')
                self.checkpoint = torch.load(resume_path)
                self.pinn.load_state_dict(self.checkpoint['state_dict'])

        # Optimizers
        params = [param for param in self.pinn.parameters() if param.requires_grad==True]
        self.optimizer_Adam  = optim.Adam(params, lr=self.lr)
        self.lr_scheduler    = StepLR(self.optimizer_Adam, step_size=2000, gamma=0.7)
        self.optimizer_LBFGS = optim.LBFGS(params, max_iter=20, tolerance_grad=1.e-8, tolerance_change=1.e-12)
        # self.optimizer_LBFGS = optim.LBFGS(params, max_iter=50, tolerance_grad=1.e-10, tolerance_change=1.e-15)

        self.pinn.train()
        self.net.to(self.device)
        self.net_pde.to(self.device)
        self.pinn.to(self.device)

        self.net.zero_grad()
        self.net_pde.zero_grad()
        self.pinn.zero_grad()

        self.writer  = SummaryWriter(comment=f'_{self.pde_case}_{self.domain_type}_{self.nn_name}_{self.z_coarse_num}_{self.args.x_coarse_num}_{self.args.x_middle_num}_{self.args.x_refine_num}_{self.scale[0]}_{self.scale[1]}_{self.blocks_num[0]}_{self.blocks_num[1]}_#{k}')
        np.savetxt(os.path.join(self.model_path, 'info.csv'), self.blocks_info, fmt='%.6e', delimiter=',')
        
        best_loss = 1.e10
        
        # Validset for block #k
        self.z_valid  = np.mean(self.blocks[k]['X_interior'][:, [2, 3]].numpy(), axis=0)[None, :]
        self.validset = Validset(self.args, self.z_valid)
        print(self.validset)
        self.X_valid  = self.validset()
            
        # Trainset for block #k
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            data = self.blocks[k]
            self.X_interior = data['X_interior'].to(self.device)
            self.X_boundary = data['X_boundary'].to(self.device)
            self.u_boundary = data['u_boundary'].to(self.device)
        
        print(f"Start Trainning (blocks #{k})")
        step = 0
        
        tt = time.time()
        output_file = open(os.path.join(f'{self.model_path}', f'output_{k}.txt'), 'w+')
        # Training Process using ADAM Optimizer
        for epoch in range(self.epochs_Adam):
            train_loss, train_loss_interior, train_loss_symmetry, train_loss_boundary = self.train_Adam(epoch)

            if (epoch + 1) % 100 == 0:
                infos = f'{self.pde_case}_{self.domain_type} ' +  \
                    f'{self.tol:.1e} ' + \
                    f'{self.sigma} '   + \
                    f'cuda{self.cuda_index} ' + \
                    f'({self.z_coarse_num} '  + \
                    f'{self.x_coarse_num}-{self.x_middle_num}-{self.x_refine_num} ' + \
                    f'{self.scale[0]}-{self.scale[1]}) ' + \
                    f'#{k}/{len(self.blocks)-1} ' + \
                    f"Adam  " + \
                    f"Epoch: {epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS:5d} " + \
                    f"loss: {train_loss:.4e} ({train_loss_interior:.4e} + {train_loss_symmetry:.4e} + {train_loss_boundary:.4e}) " + \
                    f"lr: {self.lr_scheduler.get_lr()[0]:.2e} " + \
                    f"time: {time.time()-tt:.2f}"
                print(infos)                
                tt = time.time()

            if (epoch + 1) % 1000 == 0:
                step += 1
                valid_loss = self.validate(step)
                self.pinn.train()
        
                is_best = train_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    # 'optimizer_state_dict':, self.optimizer_Adam.state_dict(), 
                    'best_loss': best_loss
                }
                
                save_checkpoints(k, state, is_best, save_dir=self.model_path)
                
            if train_loss < 5e-1:
                print(f'train_loss after Adam is {train_loss:.4e}')
                break

        train_loss_old = 1e3
        # Training Process using LBFGS Optimizer
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            train_loss, train_loss_interior, train_loss_symmetry, train_loss_boundary = self.train_LBFGS(epoch)

            if (epoch+1) % 20 == 0:
                infos = f'{self.pde_case}_{self.domain_type} ' + \
                    f'{self.tol:.1e} ' + \
                    f'{self.sigma} '   + \
                    f'cuda{self.cuda_index} ' + \
                    f'({self.z_coarse_num} ' + \
                    f'{self.x_coarse_num}-{self.x_middle_num}-{self.x_refine_num} ' + \
                    f'{self.scale[0]}-{self.scale[1]}) ' + \
                    f'#{k}/{len(self.blocks)-1} ' + \
                    f'LBFGS ' + \
                    f'Epoch: {epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS:5d} ' + \
                    f"loss: {train_loss:.4e} ({train_loss_interior:.4e} + {train_loss_symmetry:.4e} + {train_loss_boundary:.4e}) " + \
                    f'time: {time.time()-tt:.2f}'                
                print(infos)
                output_file.write(f"{train_loss:.4e} {train_loss_interior:.4e} {train_loss_symmetry:.4e} {train_loss_boundary:.4e}\n")
                tt = time.time()

            if (epoch+1) % 100 == 0:
                step += 1
                valid_loss = self.validate(step)
                self.pinn.train()
        
                is_best = train_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    # 'optimizer_state_dict': self.optimizer_LBFGS.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoints(k, state, is_best, save_dir=self.model_path)

            if train_loss < self.tol or math.isnan(train_loss): # or np.abs(train_loss-train_loss_old) < self.tol_change:                
                print(f'train_loss is {train_loss}')
                break
            train_loss_old = train_loss

        output_file.close()
        self.writer.close()


    def train_Adam(self, epoch):
        """
        Training process using Adam optimizer
        """         
        self.optimizer_Adam.zero_grad()

        # if self.resume:
        #     self.optimizer_Adam.load_state_dict(self.checkpoint)
            
        # Forward and backward propogate
        res, res_symmetry, u_boundary_pred = self.pinn(self.X_interior, self.X_boundary)
        
        
        loss_interior = self.criterion(res, torch.zeros_like(res)) 
        loss_symmetry = self.criterion(res_symmetry, torch.zeros_like(res_symmetry))
        loss_boundary = self.criterion(u_boundary_pred, self.u_boundary)
        
        loss = loss_interior + loss_symmetry + self.lam * loss_boundary
        loss.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()
        train_loss = loss.item()

        self.writer.add_scalar('train_loss', train_loss, epoch)    

        return train_loss, loss_interior.item(), loss_symmetry.item(), loss_boundary.item()

    def train_LBFGS(self, epoch):
        """
        Training process using LBFGS optimizer
        """
        # if self.resume:
        #     self.optimizer_LBFGS.load_state_dict(self.checkpoint)


        res, res_symmetry, u_boundary_pred = self.pinn(self.X_interior, self.X_boundary)
        loss_interior = self.criterion(res, torch.zeros_like(res))
        loss_symmetry = self.criterion(res_symmetry, torch.zeros_like(res_symmetry))
        loss_boundary = self.criterion(u_boundary_pred, self.u_boundary)
         
        # Forward and backward propogate
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()            

            res, res_symmetry, u_boundary_pred = self.pinn(self.X_interior, self.X_boundary)

            loss_interior = self.criterion(res, torch.zeros_like(res))
            loss_symmetry = self.criterion(res_symmetry, torch.zeros_like(res_symmetry))
            loss_boundary = self.criterion(u_boundary_pred, self.u_boundary)
            
            loss = loss_interior + loss_symmetry + self.lam * loss_boundary            
            if loss.requires_grad:
                loss.backward()
            return loss

        self.optimizer_LBFGS.step(closure)
        train_loss = closure().item()
        
        self.writer.add_scalar('train_loss', train_loss, epoch)       
        
        return train_loss, loss_interior.item(), loss_symmetry.item(), loss_boundary.item()
    
    def validate(self, step):
        """
        Validate process
        """ 
        self.net_pde.eval()
        self.net.eval()

        data = self.X_valid
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            X = data['X'].to(self.device)
            X_interior = data['X_interior'].to(self.device)
            X_boundary = data['X_boundary'].to(self.device)
            u_boundary = data['u_boundary'].to(self.device)

        res, res_symmetry = self.net_pde(X_interior)
        u = self.net(X)
        u = u.detach().cpu().numpy()
        x = X[:, :2].detach().cpu().numpy()
        
        loss = self.criterion(res, torch.zeros_like(res)) + self.criterion(res_symmetry, torch.zeros_like(res_symmetry))
        valid_loss = loss.item()

        print(f'{" "*29}valid loss: {valid_loss:.4e}')

        self.writer.add_scalar('valid_loss', valid_loss, step)
        elems = self.validset.get_x_elems()
        
        fig = show_image(x, u, elems)
        self.writer.add_figure(tag='prediction', figure=fig, global_step=step)
    
        return valid_loss


class Tester(object):
    def __init__(self, args):
        self.args         = args
        self.device       = args.device
        self.cuda_index   = args.cuda_index
        self.problem      = args.problem
        self.sigma        = args.sigma
        self.pde_case     = args.pde_case
        self.domain       = args.domain
        self.domain_type  = args.domain_type

        self.z_coarse_num = args.z_coarse_num
        self.x_coarse_num = args.x_coarse_num
        self.x_middle_num = args.x_middle_num
        self.x_refine_num = args.x_refine_num
        self.scale        = args.scale 
        self.blocks_num   = args.blocks_num
        self.model_dir = os.path.join('checkpoints',
                                      f'{self.pde_case}_{self.domain_type}',
                                      f'GFNet_{self.sigma}',
                                      f'{self.z_coarse_num}',
                                      f'{self.x_coarse_num}_{self.x_middle_num}_{self.x_refine_num}',
                                      f'{self.scale[0]}_{self.scale[1]}',
                                      f'{self.blocks_num[0]}_{self.blocks_num[1]}')
        # Network
        self.net     = Net(args.layers)
        self.net_pde = Net_PDE(self.net, self.problem, self.device)
        self.pinn    = PINN(self.net, self.problem, self.device)
        
        # Criterion
        self.criterion = nn.MSELoss()

    def _blocks_info(self):
        df = pd.read_csv(os.path.join(self.model_dir, 'info.csv'),
                         sep=',',
                         names=['x0_min', 'x0_max', 'x1_min', 'x1_max'])
        return np.hstack((df['x0_min'].values[:, None], df['x0_max'].values[:, None],
                          df['x1_min'].values[:, None], df['x1_max'].values[:, None]))

    def _get_model_index(self, z):
        """determine the block which z belongs to, and return the index of this block
        
        Params
        ======
        z: (ndarray) with shape (1, 2) 
            a 2D point
        """
        infos = self._blocks_info()
        for k in range(self.blocks_num[0]*self.blocks_num[1]):
            x0_min, x0_max, x1_min, x1_max = infos[k, 0], infos[k, 1], infos[k, 2], infos[k, 3]
            if z[0, 0] >= x0_min and z[0, 0] <= x0_max and z[0, 1] >= x1_min and z[0, 1] <= x1_max:
                return k
        
    def test(self, z):
        # datasets
        self.z = z
        index = self._get_model_index(self.z)
        
        self.testset  = Validset(self.args, self.z)
        data = self.testset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            X = data['X'].to(self.device)
            X_interior = data['X_interior'].to(self.device)
            
        model_path = os.path.join(self.model_dir, f'{index}.pth.tar')
        # load model parameters
        best_model = torch.load(os.path.join(model_path), map_location='cuda:0')
        self.pinn.load_state_dict(best_model['state_dict'])

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_pde.to(self.device)

        self.net.eval()
        self.net_pde.eval()
        self.pinn.eval()

        res, res_symmetry = self.net_pde(X_interior)
        u = self.net(X)
        
        u = u.detach().cpu().numpy()
        x = X[:, :2].detach().cpu().numpy()
        
        loss = self.criterion(res, torch.zeros_like(res))
        test_loss = loss.item()

        print(f'Test loss: {test_loss: .4e}')

        elems = self.testset.get_x_elems()
        # show_image(x, u, elems)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        tri = Triangulation(x[:, 0], x[:, 1], elems)
        surf = ax.plot_trisurf(tri, u[:, 0], cmap=plt.cm.Spectral)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$G$')
        # Add a color bar which map values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(f'gfimages/{self.args.pde_case}_{self.args.domain_type}_GF_crop.pdf',bbox_inches='tight',pad_inches=0)
        plt.show()
        
        return test_loss
        
if __name__ == '__main__':
    args = Options().parse()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.pde_case == 'Poisson':
        args.problem = Poisson(sigma=args.sigma, domain=args.domain)
    elif args.pde_case == 'DiffusionReaction':
        args.problem = DiffusionReaction(sigma=args.sigma, domain=args.domain)
        
    trainer = Trainer(args)    
    trainer.train()

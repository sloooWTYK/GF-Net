#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import os

from problem import Poisson
from utils import tile
from options import Options

class Trainset(object):
    """
    Generate DataSet for Training 

    Params
    ======
    problem (Problem)   : describe the PDE, including the boundary condition and sigma in Gaussian function
    x_coarse_path (str): file path of x sampling (coarse)
    x_middle_path (str): file path of x sampling (middle)
    x_refine_path (str): file path of x sampling (refine)
    """
    def __init__(self, args):
        if args.domain_type == 'square':
            self.x_coarse_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.x_coarse_num}', 'node.csv')
            self.x_middle_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.x_middle_num}', 'node.csv')
            self.x_refine_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.x_refine_num}', 'node.csv')
            self.z_coarse_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.z_coarse_num}', 'node.csv')
        else:
            self.x_coarse_path   = os.path.join('mesh', args.domain_type, f'{args.x_coarse_num}', 'node.csv')
            self.x_middle_path   = os.path.join('mesh', args.domain_type, f'{args.x_middle_num}', 'node.csv')
            self.x_refine_path   = os.path.join('mesh', args.domain_type, f'{args.x_refine_num}', 'node.csv')
            self.z_coarse_path   = os.path.join('mesh', args.domain_type, f'{args.z_coarse_num}', 'node.csv')            
        self.problem         = args.problem
        self.scale           = args.scale
        self.blocks_num      = args.blocks_num
        self.sigma           = self.problem.sigma
        
        self.x_coarse, self.x_coarse_mark = self._read_csv(self.x_coarse_path)
        self.x_middle, self.x_middle_mark = self._read_csv(self.x_middle_path)
        self.x_refine, self.x_refine_mark = self._read_csv(self.x_refine_path)
        self.z_blocks = self._z_blocks()

    def _read_csv(self, path):
        df = pd.read_csv(path,
                         skiprows=[0],    
                         skipfooter=1,    
                         engine='python', 
                         sep='\s+',       
                         names=['idx', 'x0', 'x1', 'mark']
        )
        return np.hstack((df['x0'].values[:, None], df['x1'].values[:, None])), df['mark'].values

    def _blocks_info(self):
        x0_min, x0_max, x1_min, x1_max = self.problem.domain
        x0 = np.linspace(x0_min, x0_max, self.blocks_num[0]+1)
        x1 = np.linspace(x1_min, x1_max, self.blocks_num[1]+1)        
        return np.array([(x0[i], x0[i+1], x1[j], x1[j+1]) for i in range(x0.shape[0]-1) for j in range(x1.shape[0]-1)])

    def _blocks_indices(self):

        indices = []
        infos = self._blocks_info()
        z = self.z_sampling()
        
        for i in range(self.blocks_num[0]*self.blocks_num[1]):
            x0_min, x0_max, x1_min, x1_max = infos[i, 0], infos[i, 1], infos[i, 2], infos[i, 3]
            mark = np.all([z[:, 0] >= x0_min,
                           z[:, 0] <= x0_max,
                           z[:, 1] >= x1_min,
                           z[:, 1] <= x1_max], axis=0)
            indice = np.where(mark == True)
            indices.append(indice[0])

        return indices
            
    def z_sampling(self):
        df = pd.read_csv(self.z_coarse_path,
                         skiprows=[0],    
                         skipfooter=1,    
                         engine='python', 
                         sep='\s+',                         
                         names=['idx', 'x0', 'x1', 'mark'])
        z, z_mark = np.hstack((df['x0'].values[:, None], df['x1'].values[:, None])),  df['mark'].values
        return z[z_mark == 0]

    def _z_blocks(self):
        z = self.z_sampling()
        blocks_indices = self._blocks_indices()
        z_blocks = []
        for i in range(len(blocks_indices)):
            if blocks_indices[i].shape[0] > 0: 
                z_blocks.append(z[blocks_indices[i]])
            else:
                z_blocks.append(None)
        return z_blocks
    
    def x_sampling(self, z):
        x_coarse_choice = np.linalg.norm(self.x_coarse - z, axis=1, ord=np.inf) >= self.scale[1]*self.sigma
        x_middle_choice = np.all([np.linalg.norm(self.x_middle - z, axis=1, ord=np.inf) >= self.scale[0]*self.sigma,
                                  np.linalg.norm(self.x_middle - z, axis=1, ord=np.inf) <= self.scale[1]*self.sigma], axis=0)
        x_refine_choice = np.linalg.norm(self.x_refine - z, axis=1, ord=np.inf) <= self.scale[0]*self.sigma

        x = np.vstack((self.x_coarse     [x_coarse_choice], self.x_middle     [x_middle_choice], self.x_refine     [x_refine_choice]))
        m = np.hstack((self.x_coarse_mark[x_coarse_choice], self.x_middle_mark[x_middle_choice], self.x_refine_mark[x_refine_choice]))

        x_interior = x[m==0]
        x_boundary = x[m>0]
        
        return x_interior, x_boundary

    def eval(self):
        
        blocks = []
        for k in range(len(self.z_blocks)):
            if self.z_blocks[k] is not None:
                z = self.z_blocks[k][[0]]
                x_interior, x_boundary = self.x_sampling(z)
                X_interior = tile(x_interior, z)
                X_boundary = tile(x_boundary, z)

                for i in range(1, self.z_blocks[k].shape[0]):
                    z = self.z_blocks[k][[i]]
                    x_interior, x_boundary = self.x_sampling(z)
                    X_interior_ = tile(x_interior, z)
                    X_boundary_ = tile(x_boundary, z)
                    X_interior  = np.vstack((X_interior, X_interior_))
                    X_boundary  = np.vstack((X_boundary, X_boundary_))
                
                u_boundary = self.problem.g(X_boundary)
                blocks.append({'X_interior': X_interior, 'X_boundary': X_boundary, 'u_boundary': u_boundary})

            else:
                blocks.append(None)
        return blocks

    def __call__(self):

        blocks = self.eval()
        for k in range(len(blocks)):
            if blocks[k] is not None:
                blocks[k]['X_interior'] = torch.from_numpy(blocks[k]['X_interior']).float()
                blocks[k]['X_boundary'] = torch.from_numpy(blocks[k]['X_boundary']).float()
                blocks[k]['u_boundary'] = torch.from_numpy(blocks[k]['u_boundary']).float()
        return blocks

    def __repr__(self):
        block_nodes = [self.z_blocks[k].shape[0] if self.z_blocks[k] is not None else 0 for k in range(len(self.z_blocks))]
        s1 = '*'*30 + ' Infos of Trainset ' + '*' *30 + '\n'
        s2 = f'x info: coarse - {self.x_coarse.shape[0]},  middle - {self.x_middle.shape[0]},  refine - {self.x_refine.shape[0]}' + '\n'
        s3 = f'z info: {len(self.z_blocks)} - {block_nodes}, ' + f'total - {sum(block_nodes)}' + '\n'
        s4 = '*'*80 + '\n'
        return s1 + s2 + s3 + s4
    
class Validset(object):
    """
    Generate Dataset for Validation
    """
    def __init__(self, args, z):
        """
        z: ndarray with shape (n, 2)
        """
        if args.domain_type == 'square':
            self.x_nodes_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.x_coarse_num}', 'node.csv')
            self.x_elems_path   = os.path.join('mesh', args.domain_type, args.mesh_type, f'{args.x_coarse_num}', 'elem.csv')
        else:
            self.x_nodes_path   = os.path.join('mesh', args.domain_type, f'{args.x_coarse_num}', 'node.csv')
            self.x_elems_path   = os.path.join('mesh', args.domain_type, f'{args.x_coarse_num}', 'elem.csv')
                        
        self.x, self.x_interior, self.x_boundary = self.x_sampling()
        self.z = z

    def get_x_nodes(self):
        df = pd.read_csv(self.x_nodes_path,
                         skiprows=[0],    
                         skipfooter=1,    
                         engine='python', 
                         sep='\s+',       
                         names=['idx', 'x0', 'x1', 'mark']
        )
        return np.hstack((df['x0'].values[:, None], df['x1'].values[:, None])), df['mark'].values

    def get_x_elems(self):
        df = pd.read_csv(self.x_elems_path,
                         skiprows=[0],    
                         skipfooter=1,    
                         engine='python', 
                         sep='\s+',       
                         names=['idx', 'i', 'j', 'k']
        )
        return np.hstack((df['i'].values[:, None]-1, df['j'].values[:, None]-1, df['k'].values[:, None]-1)).astype(np.int32)
    
    def x_sampling(self):
        x, m = self.get_x_nodes()
        x_interior = x[m == 0]
        x_boundary = x[m != 0]
        return x, x_interior, x_boundary

    def __call__(self):
        X = tile(self.x, self.z)
        X_interior = tile(self.x_interior, self.z)
        X_boundary = tile(self.x_boundary, self.z)
        u_boundary = np.zeros_like(X_boundary[:, [0]])

        X = torch.from_numpy(X).float()
        X_interior = torch.from_numpy(X_interior).float()
        X_boundary = torch.from_numpy(X_boundary).float()
        u_boundary = torch.from_numpy(u_boundary).float()

        data = {'X': X, 'X_interior': X_interior, 'X_boundary': X_boundary, 'u_boundary': u_boundary}
        return data

    def __repr__(self):
        s1 = '*'*30 + ' Infos of Validset ' + '*' *30 + '\n'
        s2 = f'x info: interior - {self.x_interior.shape[0]},  boundary - {self.x_boundary.shape[0]}' + '\n'
        s3 = f'z info: {self.z[0, :]}' + '\n'
        s4 = '*'*80 + '\n'

        return s1 + s2 + s3 + s4
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.tri import Triangulation
    from problem import Problem

    args = Options().parse()
    args.problem = Poisson(sigma=args.sigma, domain=args.domain)

    if args.domain_type == 'square':
        # trainset
        trainset = Trainset(args)
        print(trainset)
        z_blocks = trainset.z_blocks

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(z_blocks)):
            z = z_blocks[i]
            ax.scatter(z[:, [0]], z[:, [1]], s=10)
        ax.set_aspect('equal')

        x_data = np.linspace(args.domain[0], args.domain[1], args.blocks_num[0]+1)
        y_data = np.linspace(args.domain[2], args.domain[3], args.blocks_num[1]+1)
        for x in x_data:
            ax.plot([x, x], [-1, 1], c='k')
        for y in y_data:
            ax.plot([-1, 1], [y, y], c='k')

        # plt.title(r'blocks of $\xi$')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.96,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig('images/Square_Zblocks.pdf', dpi=300)
        plt.show()

    elif args.domain_type == 'washer':
        # trainset
        args.z_coarse_num = 493
        args.x_coarse_num = 224
        args.x_middle_num = 1819
        args.x_refine_num = 6981
        
        trainset = Trainset(args)
        print(trainset)
        z_blocks = trainset.z_blocks

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(z_blocks)):
            z = z_blocks[i]
            ax.scatter(z[:, [0]], z[:, [1]], s=10)
        ax.set_aspect('equal')
        
        x_data = np.linspace(args.domain[0], args.domain[1], args.blocks_num[0]+1)
        y_data = np.linspace(args.domain[2], args.domain[3], args.blocks_num[1]+1)
        for y in y_data:
            ax.plot([-1, 1], [y, y], c='k')
        for x in x_data:
            ax.plot([x, x], [-1, 1], c='k')

        # plt.title(r'blocks of $\xi$')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.96,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig('images/Washer_Zblocks.pdf', dpi=300)
        plt.show()

    elif args.domain_type == 'Lshape':
        # trainset
        # args.z_coarse_num = 423
        # args.x_coarse_num = 222
        # args.x_middle_num = 1589
        # args.x_refine_num = 6157
        
        trainset = Trainset(args)
        print(trainset)
        z_blocks = trainset.z_blocks

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for k in range(len(z_blocks)):
            if z_blocks[k] is not None:
                z = z_blocks[k]
                ax.scatter(z[:, [0]], z[:, [1]], s=10)
        ax.set_aspect('equal')

        x_data = np.linspace(args.domain[0], args.domain[1], args.blocks_num[0]+1)
        y_data = np.linspace(args.domain[2], args.domain[3], args.blocks_num[1]+1)
        for x in x_data:
            ax.plot([x, x], [-1, 1], c='k')
        for y in y_data:
            ax.plot([-1, 1], [y, y], c='k')

        # plt.title(r'blocks of $\xi$')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.96,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig('images/Lshape_Zblocks.pdf', dpi=300)
        plt.show()
                        

    if args.domain_type == 'square':
        args.x_coarse_num = 145
        args.x_middle_num = 2105
        args.x_refine_num = 8265
        trainset = Trainset(args)        
        # plot x samples related to z=(0, 0)
        z = np.array((0.0, 0.0))[None, :]
        x_interior, x_boundary = trainset.x_sampling(z)
        fig, ax = plt.subplots()
        ax.scatter(x_interior[:, [0]], x_interior[:, [1]], s=10)
        ax.scatter(x_boundary[:, [0]], x_boundary[:, [1]], s=10)
        ax.set_aspect('equal')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.96,hspace=0,wspace=0)
        plt.margins(0,0)
        # plt.title(rf'locally refined samples of $x$ for fixed $\xi$=({z[0,0]}, {z[0, 1]})')
        plt.savefig('images/Square_Xsample.pdf', dpi=300)
        plt.show()
    elif args.domain_type == 'washer':
        args.z_coarse_num = 143
        args.x_coarse_num = 224
        args.x_middle_num = 1819
        args.x_refine_num = 6981
        trainset = Trainset(args)        
        z = np.array((0.75, 0.0))[None, :]
        x_interior, x_boundary = trainset.x_sampling(z)
        fig, ax = plt.subplots()
        ax.scatter(x_interior[:, [0]], x_interior[:, [1]], s=10)
        ax.scatter(x_boundary[:, [0]], x_boundary[:, [1]], s=10)
        ax.set_aspect('equal')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0,wspace=0)
        plt.margins(0,0)
        # plt.title(rf'locally refined samples of $x$ for fixed $\xi$=({z[0,0]}, {z[0, 1]})')
        plt.savefig('images/Washer_Xsample.pdf', dpi=300)
        plt.show()
    elif args.domain_type == 'Lshape':
        # args.z_coarse_num = 143
        # args.x_coarse_num = 224
        # args.x_middle_num = 1819
        # args.x_refine_num = 6981
        trainset = Trainset(args)        
        z = np.array((-0.5, 0.0))[None, :]
        x_interior, x_boundary = trainset.x_sampling(z)
        fig, ax = plt.subplots()
        ax.scatter(x_interior[:, [0]], x_interior[:, [1]], s=10)
        ax.scatter(x_boundary[:, [0]], x_boundary[:, [1]], s=10)
        ax.set_aspect('equal')
        plt.axis('off')
        fig.set_size_inches(15.0/3.0, 15.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0,wspace=0)
        plt.margins(0,0)
        # plt.title(rf'locally refined samples of $x$ for fixed $\xi$=({z[0,0]}, {z[0, 1]})')
        plt.savefig('images/Lshape_Xsample.pdf', dpi=300)
        plt.show()

    

        # blocks = trainset.eval()
        # print(blocks)

    # # Validset
    # z = np.array((0., 0))[None, :]
    # validset = Validset(args, z)
    # data = validset()
    # print(validset)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(data['X_interior'][:, [0]], data['X_interior'][:, [1]], s=3)
    # ax.scatter(data['X_boundary'][:, [0]], data['X_boundary'][:, [1]], s=3)
    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.title(r'uniform samples of $x$ in validset')
    # plt.show()

    # # nodes = validset.get_x_nodes()[0]
    # # elems = validset.get_x_elems()
    # # tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)

    
    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # # u = np.zeros_like(nodes[:, 0])
    # # ax.plot_trisurf(tri, u, cmap=plt.cm.Spectral)
    # # plt.show()

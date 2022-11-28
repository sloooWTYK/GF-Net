#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

class Mesh(object):
    def __init__(self, base_dir='.', domain_type='Lshape', n_nodes=145):
        self.elem_path = os.path.join(base_dir, domain_type, str(n_nodes), 'elem.csv')
        self.node_path = os.path.join(base_dir, domain_type, str(n_nodes), 'node.csv')
        self.target    = os.path.join(base_dir, domain_type, str(n_nodes), 'mesh.obj')
        self.df_node   = self._df('node')
        self.df_elem   = self._df('elem')
        
    def _df(self, mode='node'):
        if mode == 'node':
            return pd.read_csv(self.node_path,
                               skiprows=[0],
                               skipfooter=1,    
                               engine='python', 
                               sep='\s+',
                               names=['index', 'x', 'y', 'mark'])
        elif mode == 'elem':
            return pd.read_csv(self.elem_path,
                               skiprows=[0],
                               skipfooter=1,    
                               engine='python', 
                               sep='\s+',
                               names=['index', 'v0', 'v1', 'v2'])
        
    def __call__(self):
        x = self.df_node['x'][:, None]
        y = self.df_node['y'][:, None]
        z = np.zeros_like(x)
        X = np.hstack((x, y, z))

        v0 = self.df_elem['v0'][:, None]
        v1 = self.df_elem['v1'][:, None]
        v2 = self.df_elem['v2'][:, None]
        V  = np.hstack((v0, v1, v2))

        print(f'number of nodes: {x.shape[0]}')
        with open(self.target, 'w') as f:
            for i in range(X.shape[0]):
                f.write(f'v {X[i, 0]} {X[i, 1]} {X[i, 2]}\n')

            for i in range(V.shape[0]):
                f.write(f'f {V[i, 0]} {V[i, 1]} {V[i, 2]}\n')

        print(f'{self.target} is generated successfully!')

    def display(self):
        os.system(f'meshlab {self.target}')
        
if __name__ == '__main__':

    domain_type = 'Lshape'
    items = os.listdir(os.path.join('.', domain_type))
    n_nodes = [int(item) for item in items]
    for n in n_nodes:
        mesh = Mesh(domain_type=domain_type, n_nodes=n)
        mesh()
    
    
    
    

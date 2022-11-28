#!/usr/bin/env python
import numpy as np
import openmesh as om
import torch
from options import Options
from model import PINN, Net, Net_GRAD
from problem import Problem, Poisson, DiffusionReaction
from utils import tile

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.tri import Triangulation


import os
import time
import pandas as pd

from trainer import Trainer
from matplotlib.pyplot import MultipleLocator
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


class GreenFormula(object):
    def __init__(self, args):
        self.args         = args
        self.problem      = args.problem
        self.layers       = args.layers
        self.device       = args.device
        self.cuda_index   = args.cuda_index        
        self.qmesh        = args.qmesh
        self.qmesh_type   = args.qmesh_type
        self.sigma        = args.sigma
        self.pde_case     = args.pde_case
        self.domain_type  = args.domain_type
        self.mesh_type    = args.mesh_type
        self.z_coarse_num = args.z_coarse_num
        self.x_coarse_num = args.x_coarse_num
        self.x_middle_num = args.x_middle_num
        self.x_refine_num = args.x_refine_num
        self.scale        = args.scale
        self.blocks_num   = args.blocks_num
        
        self.model_dir = self.get_model_dir()
        self.blocks = self._blocks()                                
                                      
        self.net = Net(self.layers)
        self.pinn = PINN(self.net, self.problem, self.device)
        self.net_grad = Net_GRAD(self.net)

        # Gauss quadratures
        self.ngs_boundary = args.ngs_boundary
        self.gs_boundary  = {1: {'wts': [2],
                                 'pts': [0]}, 
                             2: {'wts': [1, 1],
                                 'pts': [-1/np.sqrt(3), 1/np.sqrt(3)]},
                             3: {'wts': [5/9, 8/9, 5/9],
                                 'pts': [-np.sqrt(3/5), 0, np.sqrt(3/5)]}}        

        self.ngs_interior = args.ngs_interior
        self.gs_interior = {1: {'wts': [1],
                                'pts': [[1/3, 1/3]]}, 
                            3: {'wts': [1/3, 1/3, 1/3], 
                                'pts': [[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]]}, 
                            4: {'wts': [-0.5625, 0.52083333333333333, 0.520833333333333, 0.520833333333333], 
                                'pts': [[1/3, 1/3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]]}}
        
        # mesh info
        meshfile = self.get_mesh_path()
        self.mesh = self._create_mesh(meshfile)

        self.X_boundary = self._get_X_boundary()
        self.X_interior = self._get_X_interior()
        self.Z_boundary = self._get_Z_boundary()
        self.Z_interior = self._get_Z_interior()

    def get_model_dir(self):
        return os.path.join('checkpoints',
                            f'{self.pde_case}_{self.domain_type}',
                            f'GFNet_{self.sigma}',
                            f'{self.z_coarse_num}',
                            f'{self.x_coarse_num}_{self.x_middle_num}_{self.x_refine_num}',
                            f'{self.scale[0]}_{self.scale[1]}',
                            f'{self.blocks_num[0]}_{self.blocks_num[1]}')


    def get_mesh_path(self):
        if self.domain_type == 'square':
            return os.path.join('mesh', self.domain_type, self.qmesh_type, f'{self.qmesh}', 'mesh.obj')
        else:
            return os.path.join('mesh', self.domain_type, f'{self.qmesh}', 'mesh.obj')
    def _blocks(self):
        df = pd.read_csv(os.path.join(self.model_dir, 'info.csv'),
                         sep=',',
                         names=['x0_min', 'x0_max', 'x1_min', 'x1_max'])
        return np.hstack((df['x0_min'].values[:, None], df['x0_max'].values[:, None],
                          df['x1_min'].values[:, None], df['x1_max'].values[:, None]))

    def _create_mesh(self, meshfile):
        return om.read_trimesh(meshfile)

    def _face_infos(self):
        if not self.mesh.has_face_property('infos'):
            self.mesh.face_property('infos')

        for fh in self.mesh.faces():
            vertices = np.vstack([self.mesh.point(fvh) for fvh in self.mesh.fv(fh)])

            # center = np.mean(vertices, axis=0)
            area = calc_area(vertices)
            pts = [np.array([1-pt[0]-pt[1], pt[0], pt[1]] @ vertices[:, :2]) for pt in self.gs_interior[self.ngs_interior]['pts']]
            wts = [area * wt for wt in self.gs_interior[self.ngs_interior]['wts']]
            self.mesh.set_face_property('infos', fh, {'idx': fh.idx(), 'area': area, 'pts': pts, 'wts': wts})
            
    def _halfedge_infos(self):
        if not self.mesh.has_halfedge_property('infos'):
            self.mesh.halfedge_property('infos')
            
        for heh in self.mesh.halfedges():
            if self.mesh.is_boundary(heh):
                from_vertex = self.mesh.point(self.mesh.from_vertex_handle(heh))
                to_vertex = self.mesh.point(self.mesh.to_vertex_handle(heh))

                # Compute the exterior normal on each boundary halfedge
                tangent = to_vertex - from_vertex
                R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                normal = tangent @ R
                normal = normal / np.linalg.norm(normal)

                # Compute the length of each boundary halfedge
                length = np.linalg.norm(to_vertex - from_vertex)

                # Compute the center of each boundary halfedge
                a, b  = from_vertex, to_vertex
                pts = [(b - a) / 2 * pt + (a + b) / 2 for pt in self.gs_boundary[self.ngs_boundary]['pts']]
                wts = [length / 2 * wt  for wt in self.gs_boundary[self.ngs_boundary]['wts']]
                
                self.mesh.set_halfedge_property('infos', heh, {'idx': heh.idx(), 'normal': normal, 'pts': pts, 'wts': wts})

    def _get_Z_interior(self):
        idx   = np.array([vh.idx()                for vh in self.mesh.vertices() if not self.mesh.is_boundary(vh)])
        coord = np.array([self.mesh.point(vh)[:2] for vh in self.mesh.vertices() if not self.mesh.is_boundary(vh)])
        return {'idx': idx, 'coord': coord}

    def _get_Z_boundary(self):
        idx   = np.array([vh.idx()                for vh in self.mesh.vertices() if self.mesh.is_boundary(vh)])
        coord = np.array([self.mesh.point(vh)[:2] for vh in self.mesh.vertices() if self.mesh.is_boundary(vh)])
        return {'idx': idx, 'coord': coord}

    def _get_X_interior(self):
        self._face_infos()
        idx   = np.array([self.mesh.face_property('infos', fh)['idx']        for fh in self.mesh.faces()])
        coord = [np.array([self.mesh.face_property('infos', fh)['pts'][k][:2] for fh in self.mesh.faces()]) for k in range(self.ngs_interior)]
        return {'idx': idx, 'coord': coord}
    
    def _get_X_boundary(self):
        self._halfedge_infos()
        
        idx    = np.array([self.mesh.halfedge_property('infos', heh)['idx']        for heh in self.mesh.halfedges() if self.mesh.is_boundary(heh)])
        normal = np.array([self.mesh.halfedge_property('infos', heh)['normal'][:2] for heh in self.mesh.halfedges() if self.mesh.is_boundary(heh)])
        coord  = [np.array([self.mesh.halfedge_property('infos', heh)['pts'][k][:2] for heh in self.mesh.halfedges() if self.mesh.is_boundary(heh)]) for k in range(self.ngs_boundary)]
        return {'idx': idx, 'coord': coord, 'normal': normal}

    def _vertex_infos(self):
        if not self.mesh.has_halfedge_property('infos'):
            self.mesh.vertex_property('infos')    

        N_interior = self.X_interior['idx'].shape[0]
        N_boundary = self.X_boundary['idx'].shape[0]            

        # Obtain the mesh with predicted solution
        result = []
        T_vertexinfo=0
        for i in range(self.blocks.shape[0]):
            block = self.blocks[i]
            Z_block = in_subdomain(self.Z_interior, block)

            model_path = os.path.join(self.model_dir, f'{i}.pth.tar')

            if os.path.exists(model_path):            
                self.pinn.to('cuda')
                self.net.to('cuda')
                self.net_grad.to('cuda')
                self.pinn.load_state_dict(torch.load(model_path,map_location='cuda:0')['state_dict'])
            else:
                continue

            t00 = time.time()
            fG_quad = np.zeros(Z_block['coord'].shape[0])
            for k in range(self.ngs_interior):                
                INPUT_interior = tile(self.X_interior['coord'][k], Z_block['coord'])
                INPUT_interior = torch.from_numpy(INPUT_interior).float().to('cuda')
                G_interior  = self.net(INPUT_interior)
                INPUT_interior = INPUT_interior.detach().cpu().numpy()
                G_interior  = G_interior.detach().cpu().numpy()
                G_interior  = G_interior.reshape(-1, N_interior).transpose()                        
                f_interior = self.problem.f(INPUT_interior).reshape(-1, N_interior).transpose()
                fG_interior  = f_interior * G_interior                        
                
                for j in range(N_interior):
                    idx = self.X_interior['idx'][j]
                    fh = self.mesh.face_handle(idx)
                    wts = self.mesh.face_property('infos', fh)['wts']
                    fG_quad += fG_interior[j, :] * wts[k]

            gGn_quad = np.zeros_like(fG_quad)
            for k in range(self.ngs_boundary):

                INPUT_boundary = tile(self.X_boundary['coord'][k], Z_block['coord'])
                INPUT_boundary = torch.from_numpy(INPUT_boundary).float().to('cuda')            
                GX_boundary = self.net_grad(INPUT_boundary)
                INPUT_boundary = INPUT_boundary.detach().cpu().numpy()
                GX_boundary = GX_boundary.detach().cpu().numpy()
            
                Gx_boundary = GX_boundary[:, [0]].reshape(-1, N_boundary).transpose()
                Gy_boundary = GX_boundary[:, [1]].reshape(-1, N_boundary).transpose()
                nx_boundary = self.X_boundary['normal'][:, [0]]
                ny_boundary = self.X_boundary['normal'][:, [1]]
            
                Gn_boundary = Gx_boundary * nx_boundary + Gy_boundary * ny_boundary            
                g_boundary = self.problem.g(INPUT_boundary).reshape(-1, N_boundary).transpose()
                a_boundary = self.problem.a(INPUT_boundary).reshape(-1, N_boundary).transpose()
                gGn_boundary = a_boundary * g_boundary * Gn_boundary

                for j in range(N_boundary):
                    idx = self.X_boundary['idx'][j]
                    heh = self.mesh.halfedge_handle(idx)                    
                    wts = self.mesh.halfedge_property('infos', heh)['wts']
                    gGn_quad += gGn_boundary[j, :] * wts[k]
            
            quad = (fG_quad - gGn_quad)[:, None]
                

            for j in range(Z_block['idx'].shape[0]):
                idx = Z_block['idx'][j]
                vh = self.mesh.vertex_handle(idx)
                coord = self.mesh.point(vh)[None, :2]
                u_exac = self.problem.u_exact(coord).squeeze()
                u_pred = quad[j].squeeze()
                error = (u_exac - u_pred)
                self.mesh.set_vertex_property('infos', vh, {'idx': vh.idx(), 'u_exac': u_exac, 'u_pred': u_pred, 'error': error})
            T_vertexinfo +=time.time()-t00
        print(f'Time:{T_vertexinfo:.2f}')
        
    def save_obj(self):
        
        for vh in self.mesh.vertices():            
            if not self.mesh.is_boundary(vh):
                self.mesh.point(vh)[2] = self.mesh.vertex_property('infos', vh)['u_exac']
            else:
                coord = self.mesh.point(vh)[None, :2]
                self.mesh.point(vh)[2] = self.problem.u_exact(coord)
        om.write_mesh('results/u_exac.obj', self.mesh)

        for vh in self.mesh.vertices():
            if not self.mesh.is_boundary(vh):
                self.mesh.point(vh)[2] = self.mesh.vertex_property('infos', vh)['u_pred']
            else:
                coord = self.mesh.point(vh)[None, :2]
                self.mesh.point(vh)[2] = self.problem.u_exact(coord)
        om.write_mesh('results/u_pred.obj', self.mesh)

        for vh in self.mesh.vertices():
            if not self.mesh.is_boundary(vh):
                self.mesh.point(vh)[2] = self.mesh.vertex_property('infos', vh)['error']
            else:
                self.mesh.point(vh)[2] = 0.0
        om.write_mesh('results/error.obj', self.mesh)    

    def save_fig(self):
        x = np.array([self.mesh.point(vh)[:2] for vh in self.mesh.vertices()])
        elems = np.array([[fvh.idx() for fvh in self.mesh.fv(fh)] for fh in self.mesh.faces()]).astype(np.int32)
        
        u_exac = np.array([self.mesh.vertex_property('infos', vh)['u_exac'] if not self.mesh.is_boundary(vh) else self.problem.u_exact(self.mesh.point(vh)[None, :2]) for vh in self.mesh.vertices()]).astype(np.float32)
        u_pred = np.array([self.mesh.vertex_property('infos', vh)['u_pred'] if not self.mesh.is_boundary(vh) else self.problem.u_exact(self.mesh.point(vh)[None, :2]) for vh in self.mesh.vertices()]).astype(np.float32)
        error  = (u_exac - u_pred)
        
        tri = Triangulation(x[:, 0], x[:, 1], elems)
        
        fig1 = plt.figure()
        
        ax1 = fig1.add_subplot(1,1,1, projection='3d')
        surf1 = ax1.plot_trisurf(tri, u_exac, cmap=cm.coolwarm)
        fig1.colorbar(surf1, shrink=0.7,orientation='vertical')
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')
        ax1.view_init(35,-60)
        x_major_locator=MultipleLocator(0.5)
        y_major_locator=MultipleLocator(0.5)
        axxx=plt.gca()
        axxx.xaxis.set_major_locator(x_major_locator)
        axxx.yaxis.set_major_locator(y_major_locator)
        fig1.savefig(f'results/{self.args.pde_case}_{self.args.domain_type}_{self.args.solution_case}_exact.pdf',bbox_inches='tight',pad_inches=0.2)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1, projection='3d')
        surf2 = ax2.plot_trisurf(tri, u_pred, cmap=cm.coolwarm)
        fig2.colorbar(surf2, shrink=0.7,orientation='vertical')
        ax2.set_xlabel(r'$x_1$')
        ax2.set_ylabel(r'$x_2$')
        ax2.view_init(35,-60)
        x_major_locator=MultipleLocator(0.5)
        y_major_locator=MultipleLocator(0.5)
        axxx=plt.gca()
        axxx.xaxis.set_major_locator(x_major_locator)
        axxx.yaxis.set_major_locator(y_major_locator)
        fig2.savefig(f'results/{self.args.pde_case}_{self.args.domain_type}_{self.args.solution_case}_predict.pdf',bbox_inches='tight',pad_inches=0.2)

        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(1,1,1,projection='3d')
        # surf3=ax3.plot_trisurf(tri,error, cmap=cm.coolwarm)
        # fig3.colorbar(surf3, shrink=0.7,orientation='vertical')
        # ax3.set_xlabel(r'$x_0$')
        # ax3.set_ylabel(r'$x_1$')
        # ax3.view_init(35,-60)
        # x_major_locator=MultipleLocator(0.5)
        # y_major_locator=MultipleLocator(0.5)
        # axxx=plt.gca()
        # axxx.xaxis.set_major_locator(x_major_locator)
        # axxx.yaxis.set_major_locator(y_major_locator)
        # fig3.savefig(f'new_images1/{self.args.pde_case}_{self.args.domain_type}_{self.args.solution_case}_error.pdf',bbox_inches='tight',pad_inches=0.2)


        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)
        surf4=ax4.tricontourf(tri,error, cmap=cm.coolwarm)
        fig4.colorbar(surf4, shrink=0.7,orientation='vertical')
        ax4.set_xlabel(r'$x_1$')
        ax4.set_ylabel(r'$x_2$')
        # ax3.view_init(35,-60)
        x_major_locator=MultipleLocator(0.5)
        y_major_locator=MultipleLocator(0.5)
        axxx=plt.gca()
        axxx.xaxis.set_major_locator(x_major_locator)
        axxx.yaxis.set_major_locator(y_major_locator)
        fig4.savefig(f'results/{self.args.pde_case}_{self.args.domain_type}_{self.args.solution_case}_error_hm.pdf',bbox_inches='tight',pad_inches=0.2)
        # plt.savefig(f'images/{self.args.pde_case}_{self.args.domain_type}_{self.args.solution_case}.pdf')
        # plt.show()
        
    def error(self):
        err_L2 = 0.0
        denom = 0.0
        err_Linf = -np.inf
        for vh in self.mesh.vertices():
            if not self.mesh.is_boundary(vh):
                err = self.mesh.vertex_property('infos', vh)['error']
                ue  = self.mesh.vertex_property('infos', vh)['u_exac']
                # Compute err_Linf
                if err > err_Linf:
                    err_Linf = err

                # Compute err_L2
                area = 0.0
                for fh in self.mesh.vf(vh):
                    area += self.mesh.face_property('infos', fh)['area'] /3
                err_L2 += err**2 * area
                denom += ue**2 * area
        return err_Linf, np.sqrt(err_L2) / np.sqrt(denom)


if __name__ == '__main__':
    args = Options().parse()


    # args.domain_type = 'washer'
    # args.solution_case = 6
    # args.z_coarse_num = 493
    # args.x_coarse_num = 1819
    # args.x_middle_num = 6981
    # args.x_refine_num = 27352
    # args.sigma = 0.02
    # args.qmesh = 493
    # args.blocks_num = [4, 4]

    args.pde_case = 'Poisson'
    if args.pde_case == 'Poisson':
        args.problem = Poisson(sigma=args.sigma, case=args.solution_case)
    elif args.pde_case == 'DiffusionReaction':
        args.problem = DiffusionReaction(sigma=args.sigma, case=args.solution_case)

    tt = time.time()
    gf = GreenFormula(args)
    gf._vertex_infos()

    # gf.save_obj()
    
    # Compute the errors
    ttt=time.time()
    err_linf, err_l2 = gf.error()
    print(f'Time:{time.time()-ttt:.2f}')
    # print(f'{args.qmesh:4d} Linf error: {err_linf:.4e}, L2 error: {err_l2:.4e}')
    print(f'{args.qmesh:4d} & {err_linf:.2e} & {err_l2:.2e}')

    # Plot the fig of u_exact, u_pred, and error
    gf.save_fig()
    # print(f'Ellpased time is {time.time() - tt}')



    

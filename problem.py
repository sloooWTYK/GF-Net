#!/usr/bin/env python
import numpy as np

class Problem(object):
    def __init__(self, sigma=0.02, domain=(-1, 1, -1, 1), pde_case='Poisson', case=0):
        self.sigma = sigma
        self.domain = domain
        self.pde_case = pde_case
        self.case = case 

    def __repr__(self):
        return f'Poisson problem with point source with {args.sigma}'
        
    def u_exact(self, x, order=None):       
        raise NotImplementedError('Not Implemented')
    
    def f(self, x):
        """right-hand term, originally delta function, but here use Gaussian function to approximate it.
        
        Params:
        -------
        x: ndarrays of float with shape (n, 4)
        """
        if self.case == 0:
            return np.exp(-((x[:, [0]]-x[:, [2]])**2 + (x[:, [1]]-x[:, [3]])**2)/(2*self.sigma**2))/(2*np.pi*self.sigma**2)
        else:
            a     = self.a(x)
            a_1st = self.a(x, order=1)        
            a_x, a_y = a_1st[:, [0]], a_1st[:, [1]]

            u     = self.u_exact(x)
            u_1st = self.u_exact(x, order=1)
            u_2nd = self.u_exact(x, order=2)
            u_x,  u_y  = u_1st[:, [0]], u_1st[:, [1]]
            u_xx, u_yy = u_2nd[:, [0]], u_2nd[:, [1]]
            
            r = self.r(x)
        
            return -(a * (u_xx + u_yy) + (a_x * u_x + a_y * u_y)) + r * u            

    def a(self, x, order=0):
        """variable diffusion coefficients and its gradients
        """
        if self.pde_case == 'Poisson':
            if order == 0:
                return np.ones_like(x[:, [0]])
            elif order == 1:
                return np.zeros_like(x[:, :2])
        elif self.pde_case == 'DiffusionReaction':
            if order == 0:
                return 2*x[:, [1]]**2 + 1
            elif order == 1:
                return np.hstack((np.zeros_like(x[:, [0]]), 4*x[:, [1]]))
        else:
            raise NotImplementedError('Not implemented!')

    def r(self, x):
        """variable reaction coefficients
        """
        if self.pde_case == 'Poisson':
            return np.zeros_like(x[:, [0]])
        elif self.pde_case == 'DiffusionReaction':
            return x[:, [0]]**2 + 1
        else:
            raise NotImplementedError('Not implemented!')
        
    def g(self, x):
        """zero Dirichlet boundary condition
        Params:
        -------
        x: ndarrays of float with shape (n, 4)
        """
        if self.case == 0:
            return np.zeros_like(x[:, [0]])
        else:
            return self.u_exact(x)

class Poisson(Problem):
    """Poisson equation, used for computing the solution u by using quadrature rules
    
    Params
    ======
    case: (int), it provides two cases here, one is homogeneous and the other is inhomogenous
    """
    def __init__(self, sigma=0.02, domain=[-1, 1, -1, 1], case=0):
        super().__init__(sigma, domain, pde_case='Poisson', case=case)
        self.case = case
        
    def __repr__(self):
        return f'PDE type: {self.pde_case}, Solution case: {self.case}, sigma = {self.sigma}, Domain type: {self.domain}'
        
    def u_exact(self, x, order=0):
        if self.case == 1:
            if order == 0:
                return np.sin(2*np.pi*x[:, [0]])*np.sin(2*np.pi*x[:, [1]])
            elif order == 1:
                u_x = 2*np.pi*np.cos(2*np.pi*x[:, [0]])*np.sin(2*np.pi*x[:, [1]])
                u_y = 2*np.pi*np.sin(2*np.pi*x[:, [0]])*np.cos(2*np.pi*x[:, [1]])
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = -(2*np.pi)**2*np.sin(2*np.pi*x[:, [0]])*np.sin(2*np.pi*x[:, [1]])
                u_yy = -(2*np.pi)**2*np.sin(2*np.pi*x[:, [0]])*np.sin(2*np.pi*x[:, [1]])                
                return np.hstack((u_xx, u_yy))
        elif self.case == 2:
            if order == 0:
                u = 2*x[:, [0]]**2 + x[:, [1]]**2 + 1
                return u
            elif order == 1:
                u_x = 4*x[:, [0]]
                u_y = 2*x[:, [1]]
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = 4*np.ones_like(x[:, [0]])
                u_yy = 2*np.ones_like(x[:, [0]])
                return np.hstack((u_xx, u_yy))
        elif self.case == 3:
            if order == 0:
                u = np.cos(np.pi*x[:, [0]])*np.cos(np.pi*x[:, [1]])
                return u
            elif order == 1:
                u_x = np.cos(np.pi*x[:, [0]])*np.cos(np.pi*x[:, [1]])
                u_y = np.cos(np.pi*x[:, [0]])*np.cos(np.pi*x[:, [1]])
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = -np.pi**2*np.cos(np.pi*x[:, [0]])*np.cos(np.pi*x[:, [1]])
                u_yy = -np.pi**2*np.cos(np.pi*x[:, [0]])*np.cos(np.pi*x[:, [1]])
                return np.hstack((u_xx, u_yy))
        elif self.case == 4:#Lshape
            if order == 0:
                u = x[:, [0]]**2+x[:, [1]]**2
                return u
            elif order == 1:
                u_x = -444*x[:, [0]]
                u_y = -444*x[:, [1]]
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = 2*np.ones_like(x[:, [0]])
                u_yy = 2*np.ones_like(x[:, [0]])
                return np.hstack((u_xx, u_yy))
        elif self.case == 5:#square
            #Houston, P., Senior, B. and Suli, E., Sobolev Regularity Estimation for hp-Adaptive Finite Element Methods, in Numerical Mathematics and Advanced Applications (Berlin, 2003), F. Brezzi, A. Buffa, S Corsaro, and A. Murli, Eds., Springer-Verlag, pp. 619-644.
            if order == 0:
                u = np.zeros_like(x[:,[0]])
                for i in range(len(x[:,0])):
                    if x[i,0]<=0.6*(x[i,1]+1):
                        u[i] = np.cos(np.pi*x[i,1]/2)
                    else:
                        u[i] = np.cos(np.pi*x[i,1]/2)+(x[i,0]-0.6*(x[i,1]+1))**1.5
                return u
            elif order == 1:
                u_x = -444*x[:, [0]]
                u_y = -444*x[:, [1]]
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = np.zeros_like(x[:,[0]])
                u_yy = np.zeros_like(x[:,[0]])
                for i in range(len(x[:,0])):
                    if x[i,0]<=0.6*(x[i,1]+1):
                        u_xx[i] = 0
                        u_yy[i] = -(np.pi/2)**2*np.cos(np.pi*x[i,1]/2)
                    else:
                        u_xx[i]=0.75/(np.sqrt(x[i,0]-0.6*(x[i,1]+1)))
                        u_yy[i] = -(np.pi/2)**2*np.cos(np.pi*x[i,1]/2)+0.27/(np.sqrt(x[i,0]-0.6*(x[i,1]+1)))
                return np.hstack((u_xx, u_yy))
        elif self.case == 6:#washer
            #	Rice, J., Houstis, E., Dyksen, W., A Population of Linear, Second Order, Elliptic Partial Differential Equations on Rectangular Domains, Math. Comp. 36 (1981) 475-484.
            if order == 0:
                u = np.exp(-100*((x[:,[0]]+0.5)**2+(x[:,[1]]+0.5)**2))
                return u
            elif order == 1:
                u_x = -444*x[:, [0]]
                u_y = -444*x[:, [1]]
                return np.hstack((u_x, u_y))
            elif order == 2:
                u_xx = (40000*x[:,[0]]**2+40000*x[:,[0]]+9800)*np.exp(-100*((x[:,[0]]+0.5)**2+(x[:,[1]]+0.5)**2))
                u_yy = (40000*x[:,[1]]**2+40000*x[:,[1]]+9800)*np.exp(-100*((x[:,[0]]+0.5)**2+(x[:,[1]]+0.5)**2))

                return np.hstack((u_xx, u_yy))
class DiffusionReaction(Problem):
    def __init__(self, sigma=0.02, domain=[-1, 1, -1, 1], case=0):
        super().__init__(sigma, domain, pde_case='DiffusionReaction', case=case)
        self.case = case
        
    def __repr__(self):
        return f'PDE type: {self.pde_case}, Solution case: {self.case}, sigma = {self.sigma}, Domain type: {self.domain}'
        
    def u_exact(self, x, order=0):
        if self.case == 1:
            if order == 0:
                return 2*x[:, [0]]**2 + x[:, [1]]**2 + 1
            elif order == 1:
                return np.hstack((4*x[:, [0]], 2*x[:, [1]]))
            elif order == 2:
                return np.hstack((4*np.ones_like(x[:, [0]]), 2*np.ones_like(x[:, [0]])))
        if self.case == 2:
            if order == 0:
                u = np.exp(-(x[:, [0]]**2 + 2*x[:, [1]]**2 + 1))
                return u
            elif order == 1:
                u = np.exp(-(x[:, [0]]**2 + 2*x[:, [1]]**2 + 1))
                u_x = -2*x[:, [0]]*u
                u_y = -4*x[:, [1]]*u
                return np.hstack((u_x, u_y))
            elif order == 2:
                u = np.exp(-(x[:, [0]]**2 + 2*x[:, [1]]**2 + 1))
                u_x = -2*x[:, [0]]*u
                u_y = -4*x[:, [1]]*u
                u_xx = -2*u - 2*x[:, [0]]*u_x
                u_yy = -4*u - 4*x[:, [1]]*u_y
                return np.hstack((u_xx, u_yy))
        else:
            raise NotImplementedError('Not Implemented')
                
    
if __name__ == '__main__':
    problem = Poisson()
    print(problem)

    problem = DiffusionReaction(case=1)
    print(problem)

    

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:25:59 2022

@author: Pascal Sattler
"""

import numpy as np
from sympy import symbols, sympify, lambdify, exp, sin, cos, log, Pow, pprint, diff, zeros, simplify, Array, Matrix, Piecewise
from sympy.vector import CoordSys3D, express
from sympy.vector import divergence as div
from sympy.abc import t
from sympy import Heaviside



class ScalarPotential:
    
    contains_t = False
    
    def __init__(self, coords, positions, Phi_string, Phi_t = None, params = {}):
        Phi = sympify(Phi_string)
        assert len(coords) == 3
        for i in range(3):
            coords[i] = sympify(coords[i])
        
        for symbol in Phi.free_symbols:
            if str(symbol) == 't':
                self.contains_t = True
                t = symbol
        
        if self.contains_t and Phi_t is not None:
            raise ValueError('give time dependence either separate or included with position string')
                
        Phi = Phi.subs(params)
        
        if self.contains_t:
            self.Phi = lambdify((*coords, t), Phi, modules = ['numpy'])
            def Call(t):
                return self.Phi(*positions, t)
        elif Phi_t is None:
            self.Phi = lambdify(coords, Phi, modules = ['numpy'])
            self.Phi_value = self.Phi(*positions)
            def Call(t):
                return self.Phi_value
        else:
            self.Phi = lambdify(coords, Phi, modules = ['numpy'])
            Phi_t = sympify(Phi_t)
            Phi_t = Phi_t.subs(params)
            for symbol in Phi_t.free_symbols:
                if str(symbol) == 't':
                    t = symbol
                else:
                    raise ValueError('Phi_t has not defined parameters')
            self.Phi_t = lambdify(t, Phi_t, modules = ['numpy'])
            self.Phi_value = self.Phi(*positions)
            def Call(t):
                return self.Phi_t(t)*self.Phi_value
            
        self.call = Call

class VectorPotential:
    
    Cart = CoordSys3D('Cart')
    Loc = Cart.create_new('Loc', transformation = lambda x, y, z: (x, y, z))
    
    def __init__(self, A_string, A_t = None, params = {}, transformation = None):
        vec_A = sympify(A_string)
        assert len(vec_A) == 3
        vec_A = vec_A[0] * Matrix([[1],[0],[0]]) + vec_A[1] * Matrix([[0],[1],[0]]) + vec_A[2] * Matrix([[0],[0],[1]])
        
        if transformation is not None:
            self.Loc = self.Cart.create_new('Loc', transformation = transformation)
        trafo = self.Loc.transformation_to_parent()
        self.symbols = self.Loc.base_scalars()
        self.jac = zeros(3, 3)
        for i in range(3):
            for j in range(3):
                self.jac[i, j] = simplify(diff(trafo[i], self.symbols[j]))
        self.inv_jac = self.jac.inv()
        
        for symbol in vec_A.free_symbols:
            if str(symbol) == 'x':
                vec_A = vec_A.subs({symbol : trafo[0]})
            elif str(symbol) == 'y':
                vec_A = vec_A.subs({symbol : trafo[1]})
            elif str(symbol) == 'z':
                vec_A = vec_A.subs({symbol : trafo[2]})
        
        vec_A = vec_A.subs({self.symbols[0] : 'q1', self.symbols[1] : 'q2', self.symbols[2] : 'q3'})
        self.inv_jac = self.inv_jac.subs({self.symbols[0] : 'q1', self.symbols[1] : 'q2', self.symbols[2] : 'q3'})
        
        self.vec_A = simplify(self.inv_jac * vec_A)
        self.vec_A = self.vec_A.subs(params)
        #pprint(vec_A)
        
        self.A_t = A_t
        
    def get_A(self, t):
        t_dep = sympify(self.A_t)
        assert len(t_dep) == 3
        t_dep = t_dep[0] * Matrix([[1],[0],[0]]) + t_dep[1] * Matrix([[0],[1],[0]]) + t_dep[2] * Matrix([[0],[0],[1]])
        return self.vec_A * t_dep
    
    def get_div_A(self, t):
        pass
    
    def get_A_sqr(self, t):
        pass
    

'''    
#VectorPotential('[x, y, z]', '[t, 1, t/2]', transformation = 'spherical') #lambda rho, phi, z : (rho * cos(phi), rho * sin(phi), z))
xs = np.linspace(0, np.pi)
ys = np.linspace(0, 0, 1)
zs = np.linspace(0, 0, 1)
xx, yy, zz = np.meshgrid(xs, ys, zs)
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
phi = ScalarPotential(['x', 'y', 'z'], (xx, yy, zz), "sin(x)**2*t")

import matplotlib.pyplot as plt
plt.plot(phi.call(0.1))
'''

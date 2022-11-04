# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:06:35 2022

@author: Pascal Sattler
"""

import numpy as np

class periodic_pulay:
    
    def __init__(self, size, alpha = 0.05, n = 5, k = 3):
        self.alpha = alpha
        self.n = n
        self.k = k
        self.R = np.zeros((size, self.n))
        self.F = np.zeros((size, self.n))
        self.i = 0
        self.f_old = np.zeros(size)
        #self.rho_old = np.zeros(size)
        
    def __call__(self, rho, f):
        rho_new = rho + self.alpha*f
        if self.i < self.n:
            self.F[:,self.i] = f - self.f_old
            self.R[:,self.i] = rho_new - rho
        elif (self.i + 1)%self.k != 0:
            self.F[:,:-1] = self.F[:,1:]
            self.R[:,:-1] = self.R[:,1:]
            self.F[:,-1] = f - self.f_old
            self.R[:,-1] = rho_new - rho
        else:
            transF = self.F.T
            nonlinear =(self.R + self.alpha * self.F).dot(np.linalg.inv(transF.dot(self.F))).dot(transF)
            rho_new = np.abs(rho_new - 0*nonlinear.dot(f))
            self.F[:,:-1] = self.F[:,1:]
            self.R[:,:-1] = self.R[:,1:]
            self.F[:,-1] = f - self.f_old
            self.R[:,-1] = rho_new - rho
            
        self.f_old = f.copy()
        self.i += 1
        return rho_new
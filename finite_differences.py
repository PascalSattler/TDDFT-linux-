# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:07:00 2022

@author: Pascal Sattler
"""

import numpy as np
from scipy import linalg as lin
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

a = 1 # Potentialtopfbreite
n = 1000  # Anzahl der Punkte
n_elec = 10

w = 100

fix = True

if fix:
    h = a/(n+1)
    x = np.linspace(h, a-h, n)
    T = - (0.5 / h**2) * diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
else:
    h = a/n
    x = np.linspace(0, a, n, endpoint = False)
    T = - (0.5 / h**2) * diags([1,1, -2, 1,1], [1-n,-1, 0, 1,n-1], shape=(n, n))

'''
T = np.zeros([n,n])

for i in range(n):
    T[i,i] = -2
    if i-1 >= 0:
        T[i, i - 1] = 1
    if i+1 < n:
        T[i, i + 1] = 1 
'''
V = diags([0.5 * w**2 * (x-a/2)**2], [0], shape=(n, n))

H = T+V

E, Psi = eigsh(H, which = 'SA', k = n_elec)
# E_ex = np.pi**2/2 * np.arange(10)**2

print(E)

for i in range(4):
   plt.plot(x, (1/np.sqrt(h))*np.real(Psi[:,i]))

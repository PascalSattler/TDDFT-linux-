# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:39:50 2022

@author: Pascal Sattler
"""

import numpy as np
import matplotlib.pyplot as plt
#from DFT_classes import Hamiltonian, SoftCoulomb, TimePropagation
#from wave_function_class import WaveFunction
from scipy.fft import fftfreq, fft

omega = 3
n_opt = 4
pulse_time = (2*np.pi/omega)*n_opt

base_freq = omega/(2*np.pi)

values = np.loadtxt('dipole_values.txt', skiprows = 1)

times = values[:,0]
dip_list = values[:,1]

dt = times[1] - times[0]

accel = np.diff(dip_list, n = 2, prepend = dip_list[0])/dt**2

freqs = 2* np.pi * fftfreq(len(accel), d = dt)#/base_freq
accel_fft = fft(accel)

sorted_ind = np.argsort(freqs)
freqs = freqs[sorted_ind]
accel_fft = accel_fft[sorted_ind]

plt.title("FT of dipole moment acceleration")
plt.xlabel("w")
plt.grid(True)
plt.xlim(0, 30)#np.max(freqs))
plt.plot(freqs, np.log(np.abs(accel_fft)**2))
#plt.legend("best")
plt.savefig("dipole_moment_accel_fft.png")
plt.close()
#plt.show()
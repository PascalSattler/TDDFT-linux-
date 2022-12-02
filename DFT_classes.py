# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:38:54 2022

@author: Pascal Sattler
"""

import numpy as np
from scipy import linalg as lin
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.ndimage import convolve1d as conv
from scipy.optimize import root
from scipy.integrate import ode, trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#from periodic_pulay_class import periodic_pulay
from wave_function_class import WaveFunction
from electric_potential_class import ScalarPotential
from tools import correlation_pot, callback

def SoftCoulomb(x, x0):
    return 1/np.sqrt(1+(x - x0)**2)
    
def extrapolate(val, xrange, yrange):
    interp = interp1d(xrange, yrange, fill_value = 'extrapolate')
    return interp(val)

class Hamiltonian:
    
    Psi = None
    root_method = 'anderson'
    V_phi_list = []
    
    def __init__(self, n, n_elec, xrange, temp = 0, fix=True):
        self.n = n
        self.n_elec = n_elec
        assert hasattr(xrange, "__len__")
        assert len(xrange) == 2
        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.x_span = self.xmax - self.xmin
        self.fix = fix
        
        if fix:
            self.h = self.x_span/(n+1)
            self.x = np.linspace(self.xmin + self.h, self.xmax - self.h, self.n)
        else:
            self.h = self.x_span/n
            self.x = np.linspace(self.xmin, self.xmax, self.n, endpoint = False)
            
        if temp == 0:
            if n_elec % 2 == 0:
                self.f_occ = 2 * np.ones(n_elec // 2)
                #for i in range(6):
                #    self.f_occ = np.append(self.f_occ, 1e-10)
            else:
                #self.f_occ = 2 * np.ones(n_elec // 2)
                #self.f_occ = np.append(self.f_occ, 1e-10)
                self.f_occ = 2 * np.ones(n_elec // 2)
                self.f_occ = np.append(self.f_occ, 1)
        else:
            raise NotImplementedError("noch nicht fertig für endliche Temp")
            
        self._create_kinetic()
        
        self.sc_range = np.arange(-n, n + 1) * self.h
        self.kernel = SoftCoulomb(self.sc_range, 0)
        self.corr_pot = np.empty(self.n)
    
    def _create_kinetic(self):
        if self.fix:
            self.T = - (0.5 / self.h**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.n, self.n))
        else:
            self.T = - (0.5 / self.h**2) * diags([1,1, -2, 1,1], [1-self.n,-1, 0, 1,self.n-1], shape=(self.n, self.n))
            
    def create_ex_potential(self, func):
        self.V_ex = func(self.x)
        
    def couple_scal_pot(self, func_string, Phi_t = None, params = {}):
        if self.fix == True:
            self.V_phi_list = np.append(self.V_phi_list, ScalarPotential(['x', 'y', 'z'],
                (self.x, np.zeros_like(self.x), np.zeros_like(self.x)),
                func_string, Phi_t = Phi_t, params = params))
        elif self.fix == False:
            radius = (self.xmax - self.xmin)/(2*np.pi)
            angles = self.x/radius
            if func_string == 'x':
                self.V_phi_list = np.append(self.V_phi_list, ScalarPotential(['x', 'y', 'z'],
                    (radius * np.cos(angles), np.zeros_like(angles), np.zeros_like(angles)),
                    func_string, Phi_t = Phi_t, params = params))
            elif func_string == 'y':
                self.V_phi_list = np.append(self.V_phi_list, ScalarPotential(['x', 'y', 'z'],
                    (np.zeros_like(angles), radius * np.sin(angles), np.zeros_like(angles)),
                    func_string, Phi_t = Phi_t, params = params))

    def solve(self, num_eig=None):
        self.V_phi = 0 
        #for V_phi in self.V_phi_list:
        #        self.V_phi += V_phi.call(0)
        if self.Psi is None:
            self.H = self.T + diags(self.V_ex)
        else:
            self.H = self.T + diags(self.V_ex + self.correlation_pot(0) + self.hartree_pot() / 2 + self.V_phi)
            
        if num_eig is None:
            self.E, self.Psi = eigsh(self.H, which = 'SA', k = len(self.f_occ), ncv = max(4*self.n_elec + 1, 40))
        else:
            self.E, self.Psi = eigsh(self.H, which = 'SA', k = num_eig)
                
        self.Psi = WaveFunction(self.Psi/np.sqrt(self.h))
        self.prob_density = self.get_probability(self.Psi.to_array())
        #self.dummy = np.zeros_like(self.prob_density)
        
    def plot(self):
        print(self.E)
        psi = self.Psi.to_array()
        for i in range(self.Psi.n_elec):
            plt.plot(self.x, np.real(psi[:,i]))
        plt.close()
           
    def get_probability(self, Psi):
        return np.einsum('ji,j', np.abs(Psi)**2, self.f_occ)
    
    def get_probabilityReIm(self, Psi, size):
        return np.einsum('ji,j', Psi[:size]**2 + Psi[size:]**2, self.f_occ)
    
    def probability(self, psi):
        rho = np.zeros(self.n)
        for i in range(len(self.f_occ)):
            rho += self.f_occ[i] * (psi[2*i*self.n:(2*i+1)*self.n]**2 \
                                    + psi[(2*i+1)*self.n:(2*i+2)*self.n]**2)
        return rho
        #return probability(self.n, self.n_elec, self.f_occ, psi, self.dummy)
    
    #def _get_wigner_seitz(self):
    #    self.r_s = np.abs(1/(2 * self.prob_density))
    
    def correlation_pot(self, pol):
        return correlation_pot(pol, self.n, self.prob_density, self.corr_pot)
    
    def hartree_pot(self):
        return np.zeros_like(self.prob_density)
        if self.fix:
            return self.h * conv(self.prob_density, self.kernel, mode = 'constant', cval = 0)
        else:
            return self.h * conv(self.prob_density, self.kernel, mode = 'wrap')
    
    def solve_sc(self):
        if self.Psi is None:
            self.solve()
        N_it = root(self.iteration, self.prob_density, method = self.root_method,
                    tol = 1e-7, callback = callback, options = {'line_search' : 'wolfe'}).nit
        print("{} iterations were performed for convergence.".format(N_it))

    def iteration(self, rho):
        self.prob_density = rho
        self.solve()
        return self.prob_density - rho
    


class TimePropagation:

    def __init__(self, hamiltonian: Hamiltonian , psi_start: WaveFunction, t_cutoff : float = 1e16):
        self.hamiltonian = hamiltonian
        self.psi_start = psi_start
        self.t_cutoff = t_cutoff
        self.size = self.hamiltonian.n
        self.H_R = self.hamiltonian.T
        self.H_I = csr_matrix((self.size, self.size), dtype = np.float64)
        
    def _separateReImVec(self, vec):
        return np.concatenate((vec.real, vec.imag))
        
    def _get_H_time_t(self, t):
        '''
        self.V_phi = - self.hamiltonian.V_phi.call(t)
        
        self.V_ex = self.hamiltonian.V_ex
        self.V_xc = self.hamiltonian.correlation_pot(0)
        self.V_H = self.hamiltonian.hartree_pot()
        '''
        if t < self.t_cutoff:
            self.V_phi = 0 
            for V_phi in self.hamiltonian.V_phi_list:
                self.V_phi += V_phi.call(t)
            return self.hamiltonian.V_ex \
                    + self.hamiltonian.correlation_pot(0) \
                     - self.V_phi
        else:
            return self.hamiltonian.V_ex \
                    + self.hamiltonian.correlation_pot(0) \
                    


    def psi_dt(self, t, psi):
        out = np.empty_like(psi)
        V_pot = self._get_H_time_t(t)
        for i in range(self.psi_start.n_elec):
            psi_R = psi[2*i*self.size:(2*i+1)*self.size]
            psi_I = psi[(2*i+1)*self.size:(2*i+2)*self.size]
            out[2*i*self.size:(2*i+1)*self.size] =   self.H_R.dot(psi_I) + V_pot * psi_I + self.H_I.dot(psi_R) #(self.V_ex + self.V_xc + self.V_H + self.V_phi)
            out[(2*i+1)*self.size:(2*i+2)*self.size] = - self.H_R.dot(psi_R) - V_pot * psi_R + self.H_I.dot(psi_I)
        return out
    
    def _to_array(self, psi):
        psi_array = np.reshape(psi, (len(self.hamiltonian.f_occ), 2, self.size))
        return (psi_array[:,0] + 1j * psi_array[:,1])

    def time_prop(self, times):
        self.times = times
        #print(1/(self.hamiltonian.E[-1] - self.hamiltonian.E[0]))
        prop = ode(self.psi_dt).set_integrator('dop853', rtol = 1e-6, atol = 1e-6, nsteps = 1e8)#, first_step = 0.1)
        prop.set_initial_value(self.psi_start.psi.copy(), self.times[0])
        it = 1
        
        dip = dipole(self.hamiltonian.x, self.hamiltonian.fix)

        psi_list = np.empty((len(self.times), len(self.hamiltonian.f_occ), self.size), dtype = np.complex128)
        psi_list[0] = self.psi_start.to_array()

        self.rho_list = np.empty((len(self.times), self.size), dtype = np.float64)
        self.rho_list[0] = self.psi_start.probability(self.hamiltonian.f_occ)
        
        self.dip_list = np.empty(len(self.times), dtype = np.complex128)
        self.dip_list[0] = dip.call(self.rho_list[0])

        file = open('dipole_values.txt', mode = 'w')
        file.write('time \t dipole_real \t dipole_imag \n')
        #import cProfile
        #from pstats import SortKey
        #import pstats
        #import io

        while prop.successful() and prop.t < self.times[-1]:
            #pr = cProfile.Profile()
            #pr.enable()
            prop.integrate(self.times[it])
            #pr.disable()
            #s = io.StringIO()
            #sortby = SortKey.CUMULATIVE
            #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #ps.print_stats()
            #print(s.getvalue())
            #exit()

            psi_list[it] = self._to_array(prop.y)
            self.rho_list[it] = self.hamiltonian.probability(prop.y)
            #psi_t = prop.y[:self.size] + 1j * prop.y[self.size:]
            
            self.dip_list[it] = dip.call(self.rho_list[it])
            print(prop.t, self.dip_list[it])
            
            to_write_in_file = str(prop.t) + '\t' + str(self.dip_list[it].real) + '\t' + str(self.dip_list[it].imag) + '\n'
            file.write(to_write_in_file)

            #self.rho_extrap = extrapolate(self.times[it+1], self.times[:it], self.rho_list[:it])
            it += 1
        
    
        file.close()
        return psi_list
    
    def _animate(self, i):
        self.line.set_ydata(self.rho_list[i])
        return self.line,
    
    def plot_prob(self):
        fig, ax = plt.subplots()
        xrange = self.hamiltonian.x
        ax.set_ylim(0, 1.1 * np.max(self.rho_list))
        self.line, = ax.plot(xrange, self.rho_list[0])
        animation = anim.FuncAnimation(fig, self._animate, interval = 20, frames = self.rho_list.shape[0],
                                            blit = True, save_count = 50)
        animation.save("prob_density.gif", fps = 60)
        plt.show()
        
    def plot_dip(self):
        plt.title("Dipole moment")
        plt.xlabel("x")
        plt.grid(True)
        plt.plot(self.times, self.dip_list.real/self.hamiltonian.n_elec, label = "Re(dip)")
        plt.plot(self.times, self.dip_list.imag/self.hamiltonian.n_elec, label = "Im(dip)")
        plt.legend("best")
        plt.savefig("dipole_moment.pdf")
        plt.close()
        #plt.show()

    def plot_dip_accel(self, freq, excit_freq):
        accel = dipole.accel(self.dip_list, freq)
        plt.title("FT of dipole moment acceleration")
        plt.xlabel("w/w_excitation")
        plt.grid(True)
        plt.plot(freq/excit_freq, accel.real)
        plt.legend("best")
        plt.savefig("dipole_moment_accel_fft.pdf")
        plt.close()
        #plt.show()



class dipole:
    
    def __init__(self, x, fix = True):
        self.fix = fix
        if fix:
            dx_l = x[1] - x[0] 
            dx_r = x[-1] - x[-2]
            self.x = np.concatenate(([x[0] - dx_l], x, [x[-1] + dx_r]))
            self.call = self.dip_fix
        else:
            self.x = np.concatenate((x, [2*x[-1] - x[-2]]))
            self.L = self.x[-1] - self.x[0]
            self.call = self.dip_nfix
    
    def dip_fix(self, rho):
        integrand = np.concatenate(([0], rho, [0])) * self.x
        return trapezoid(integrand, self.x)
    
    def dip_nfix(self, rho):
        integrand = np.concatenate((rho, [rho[0]])) * np.exp(2j*np.pi*self.x/self.L)
        return trapezoid(integrand, self.x)


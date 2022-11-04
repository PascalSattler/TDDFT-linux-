import numpy as np
from numba import jit, int32


@jit(nopython = True, cache = True)#np.float64(int32, np.float64), nopython = True)
def correlation_pot(pol, size, prob_density, corr_pot):
    if pol == 0:
        A = 18.4029
        B = 0.0
        C = 7.50139
        D = 0.101855
        E = 0.01282710
        alpha = 1.51124
        beta = 0.2586
        exponent = 4.42425
    elif pol == 1:
        A = 5.2479
        B = 0.0
        C = 1.56823
        D = 0.1286150
        E = 0.0032074
        alpha = 0.053882
        beta = 1.56E-5
        exponent = 2.95899
    else:
        raise ValueError("Polarization must be 0 or 1!")
    
    for i in range(size):
        if prob_density[i] < 1e-16:
            corr_pot[i] = 0.0
        else:
            r_s = 1 / (2 * prob_density[i])
    
            fraction = (r_s + E*r_s**2)/(A + B*r_s + C*r_s**2 + D*r_s**3)
            logarithm = np.log(1 + alpha*r_s + beta*r_s**exponent)
            e_corr = -0.5 * fraction * logarithm

            fraction_derivative = (A + 2*A*E*r_s + (B*E-C)*r_s**2 - 2*D*r_s**3 - D*E*r_s**4 )/(A + B*r_s + C*r_s**2 + D*r_s**3)**2
            logarithm_derivative = (alpha + exponent*beta*r_s**(exponent-1))/(1 + alpha*r_s + beta*r_s**exponent)
            e_corr_derivative = r_s**2 * (fraction_derivative * logarithm + fraction * logarithm_derivative)

            corr_pot[i] = e_corr + prob_density[i] * e_corr_derivative
    
    return corr_pot
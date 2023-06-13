def jxx_terms(nmax=5, maxkrsqr=15, maxm=7, ngrid=300):
    '''
    generatie approximation for n*n*In/x for each n

    cold plasma contribution (a part of n=1/-1) is subtracted

    '''
    def func_nnIn_x(x, n=1):
        if abs(x) < 0.001:
            x = 0.001
        return -n*n*ive(n, x)/x

    from petram.phys.common.rational_approximation import find_decomposition

    L = np.linspace(0, maxkrsqr, ngrid)
    fits = [None]*nmax
    
    for i in range(nmax):
        fit = find_decomposition(func_nnIn_x, L, mmax_min=3, n = i+1)
        fit.n = i+1
        fits[i] = fit

    # subtract cold plasma part
    fits[0].c0 =  fits[0].c0 - 1

    return fits


from petram.phys.phys_const import epsilon0 as e0
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import mass_electron as me

from petram.phys.numba_coefficient import NumbaCoefficient
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff

def build_coefficients(ind_vars, omega, B, dens, temp, mass, charge, fits, g_ns, l_ns):

    B_coeff = VCoeff(3, [B], ind_vars, l, g,
                     return_complex=False, return_mfem_constant=True)
    dens_coeff = SCoeff([dens, ], ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)
    t_coeff = SCoeff([temp, ], ind_vars, l, g,
                     return_complex=False, return_mfem_constant=True)

    dependency = (B_coeff, dens_coeff, t_coeff)
    dependency = [(x.mfem_numba_coeff if isinstance(B_coeff, NumbaCoefficient) else x)
                  for x in dependency]
    
    def c0inv(ptx, B, dense, temp):
        '''
        c0_inv:
            (1 - (n*omega_x/omega)^2)/c0
        '''
        return  (1 - (n*omega_x/omega)^2)/c0

    def ccoeff(ptx, B, dens, temp):    
        '''
        cterm : 
            -1j*e0*wpx2* <<<c>>> /w / ((1 - (n*omega_e/w)^2)

        '''
        T = temp*q_base
        vTe  = sqrt(2*T/mass)

        '''

        ni  = ne *(1. - imp_frac)*has_ion
        nim = ne * imp_frac*has_ion
        LAMBDA = 1+12*pi*(e0*Te)**(3./2)/(q**3 * sqrt(ne))

        nu_ei = (qi**2 * qe**2 * ni *
           log(LAMBDA)/(4 * pi*e0**2*me**2)/vTe**3)
        nu_eim = (qim**2 * qe**2 * nim *
           log(LAMBDA)/(4 * pi*e0**2*me**2)/vTe**3)
        me_eff  = (1 -nu_ei / 1j/w - nu_eim/1j/w )*me
        wpe2  = ne * q**2/(me_eff*e0)
        '''
        wpx2  = dense * q**2/(mass*e0)
        Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_x = charge * Bnorm / mass

        return -1j*e0*wpx2*ccc/omega 

    def dcoeff(ptx, B, dens, temp):        
        '''
        dcoeff:
             <<<d>>> * (1 - (n*omega_x/omega)^2)
        '''
        Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_x = charge * Bnorm / mass
            
        return ddd * (1 - (n*omega_x/omega)^2)            

    def kappa(ptx, B, dens, temp):
        '''
        kappa: 
           -rho_sq/2 * (1 - (n*omega_x/omega)^2)

        '''
        Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_x = charge * Bnorm / mass
        rho_sq = 2*temp*q_base/mass/omega_x/omega_x
        kappa =  -rho_sq/2*(1 - (n*omega_x/omega)^2)

        return kappa

    numba_debug = False if myid != 0 else get_numba_debug()
    
    params = {'omega': omega, 'mass': mass, 'charges': charge}        
    jitter = mfem.jit.vector(shape=(3, ), complex=True, params=params,
                             debug=numba_debug, dependency=dependency)
          
    for fit in fits:
        params['n'] = fit.n
        

              





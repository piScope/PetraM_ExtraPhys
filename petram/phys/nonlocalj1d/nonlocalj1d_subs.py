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
        fits[i] = fit

    # subtract cold plasma part
    fits[0].c0 =  fits[0].c0 - 1
    return fits


from petram.phys.phys_const import epsilon0 as e0
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import mass_electron as me

def build_coefficients(ind_vars, omega, B, dens, temp, mass, charge, g_ns, l_ns):

    wp2 = dens * q_base**2/(mass_eff*e0)
    wc = qe * Bnorm/mass_eff

    @njit(complex128(float64[:], float64, float64)
    def cterm(B, dense, temp):
        '''
        -1j*e0*wpx2*c_term/w / ((1 - (n*omega_e/w)^2)

        external constant: ccc, omega
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

    @njit(complex128(float64[:])
    def dterm(B):
        '''
        dterm * (1 - (n*omega_x/omega)^2)

        external constant: ddd, n, omega
        '''
        Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_x = charge * Bnorm / mass
            
        return ddd * (1 - (n*omega_x/omega)^2)            

    @njit(complex128(float64[:], float64))
    def kappa(B, temp):
        '''
        -rho_sq/2 * (1 - (n*omega_x/omega)^2)

        external constant: ddd, n, omega
        '''
        Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_x = charge * Bnorm / mass
        rho_sq = 2*temp*q_base/mass/omega_x/omega_x
        return -rho_sq/2*(1 - (n*omega_x/omega)^2)
              





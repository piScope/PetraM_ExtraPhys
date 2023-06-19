'''

compute non-local current correction.

'''
from petram.phys.nonlocalj1d.nonlocalj1d_model import NonlocalJ1D_BaseDomain
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jxx')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('B', VtableElement('bext', type='array',
                            guilabel='magnetic field',
                            default="=[0,0,0]",
                            tip="external magnetic field")),
        ('dens', VtableElement('dens_e', type='float',
                               guilabel='density(m-3)',
                               default="1e19",
                               tip="electron density")),
        ('temperature', VtableElement('temperature', type='float',
                                      guilabel='temp.(eV)',
                                      default="10.",
                                      tip="temperature ")),
        ('mass', VtableElement('mass', type="float",
                               guilabel='masses(/Da)',
                               default="1.0",
                               # no_func=True,
                               tip="mass. normalized by Da. For electrons, use q_Da")),
        ('charge_q', VtableElement('charge_q', type='float',
                                   guilabel='charges(/q)',
                                   default="1",
                                   no_func=True,
                                   tip="charges normalized by q(=1.60217662e-19 [C])")),
        ('frac_collisions', VtableElement('frac_collisions', type='float',
                                   guilabel='alpha',
                                   default="0.0",
                                   tip="parameter for addi dampingimaginary part of omega (w -> w + jnu))")),
        ('nmax', VtableElement('nmax', type='int',
                               guilabel='nmax',
                               default="3",
                               no_func=True,
                               tip="maximum number of cyclotron harmonics ")),
        ('kprsqr_max', VtableElement('kprsqr_max', type='int',
                               guilabel='max (kp*rho)^2',
                               default="15",
                               no_func=True,
                               tip="maximum (k_perp * rho)^2 to fit the dispersion curve.")),)


class NonlocalJ1D_Jxx(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jxx, self).__init__(**kwargs)

    def count_x_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxterms = 0
            self._nmax_bk = -1

        B, dens, temp, masse, charge, fcols, nmax, kpr2max = self.vt.make_value_or_expression(
            self)

        from petram.phys.nonlocalj1d.nonlocalj1d_subs import jxx_terms

        if self._nmax_bk != nmax:
            fits = jxx_terms(nmax=nmax, maxkrsqr=kpr2max)
            self._approx_computed = True
            total = np.sum([len(fit.c_arr)+1 for fit in fits])
            self._nxterms = total
            self._nmax_bk = nmax

        return int(self._nxterms)

    def get_jx_names(self):
        base = self.get_root_phys().extra_vars_basex
        return [base + self.name() + str(i+1)
                for i in range(self.count_x_terms())]

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]

        freq, omega = em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, fcols, nmax, kpr2max = self.vt.make_value_or_expression(
            self)

        from petram.phys.nonlocalj1d.nonlocalj1d_subs import (jxx_terms,
                                                              build_coefficients)

        fits = jxx_terms(nmax=nmax, maxkrsqr=kpr2max)
        self._jitted_coeffs = build_coefficients(ind_vars, omega, B, dens, temp, mass, charge,
                                                 fcols, fits, self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        if check == 3:
            return True

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        Jnl = Jn1 + Jx2 + Jx3 ..
        nabla^2 Jx1 - d1 = c1 * E
        nabla^2 Jx1 - d2 = c2 * E
        nabla^2 Jx3 - d1 = c3 * E
        '''
        Jnlxname = self.get_root_phys().extra_vars_basex
        Jnlterms = self.get_jx_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Ename = var_s[0]

        loc = []
        for n in Jnlterms:
            loc.append((n, Ename, 1, 1))
        for n in Jnlterms:
            loc.append((Ename, n, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        jxnames = self.get_jx_names()

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)
        idx = jxnames.index(dep_var)

        if real:
            dprint1(
                "Add diffusion and mass integrator contribution(real)", dep_var, idx)
        else:
            dprint1(
                "Add diffusion and mass integrator contribution(imag)", dep_var, idx)

        f_coeffs = self._jitted_coeffs[0]
        kappa, cc, dd = f_coeffs[idx]

        if dd is not None:
            self.add_integrator(engine, 'diffusion', kappa, a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'mass', dd, a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:  # constant term contribution
            self.add_integrator(engine, 'mass', kappa, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):
        if real:
            dprint1("Add mixed contribution(real)"  "r/c", r, c, is_trans)
        else:
            dprint1("Add mixed contribution(imag)"  "r/c", r, c, is_trans)

        jxnames = self.get_jx_names()
        jnlxname = self.get_root_phys().extra_vars_basex

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]
        var_s = em1d.dep_vars
        Ename = var_s[0]

        if c == Ename:
            idx = jxnames.index(r)
            a, ccoeff, c = self._jitted_coeffs[0][idx]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        if r == Ename:
            if not real:  # -j omega
                omega = 2*np.pi*em1d.freq
                sc = mfem.ConstantCoefficient(-omega)
                self.add_integrator(engine, 'jcontribution', sc,
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            if real:  #  alpha J
                #omega = 2*np.pi*em1d.freq
                #sc = mfem.ConstantCoefficient(omega*0.005)
                alpha = self._jitted_coeffs[1]
                self.add_integrator(engine, 'jcontribution', alpha, 
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

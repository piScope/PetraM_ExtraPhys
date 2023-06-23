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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jperp')

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
                               guilabel='mass(/Da)',
                               default="1.0",
                               # no_func=True,
                               tip="mass. normalized by Da. For electrons, use q_Da")),
        ('charge_q', VtableElement('charge_q', type='float',
                                   guilabel='charge(/q)',
                                   default="1",
                                   no_func=True,
                                   tip="charges normalized by q(=1.60217662e-19 [C])")),
        ('frac_collisions', VtableElement('frac_collisions', type='float',
                                          guilabel='alpha',
                                          default="0.0",
                                          tip="additional damping due to non-local current(sigma*Jhot)")),
        ('ky', VtableElement('ky', type='float',
                             guilabel='ky',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the y direction")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the z direction")),)


class NonlocalJ1D_Jperp(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jperp, self).__init__(**kwargs)

    def _count_perp_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxyterms = 0
            self._nmax_bk = -1
            self._kprmax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)
        B, dens, temp, masse, charge, alpha, ky, kz = self.vt.make_value_or_expression(
            self)
        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin

        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import jperp_terms

        if self._nmax_bk != nmax or self._kprmax_bk != kprmax or self._mmin_bk != mmin:
            fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin)
            self._approx_computed = True
            total = 1 + len(fits[0].c_arr)
            self._nperpterms = total
            self._nmax_bk = nmax
            self._kprmax_bk = kprmax
            self._mmin_bk = mmin

        return int(self._nperpterms)

    def get_jx_names(self):
        base = self.get_root_phys().extra_vars_basex
        return [base + self.name() + str(i+1)
                for i in range(self.count_x_terms())]

    def get_jy_names(self):
        base = self.get_root_phys().extra_vars_basey
        return [base + self.name() + str(i+1)
                for i in range(self.count_y_terms())]

    def count_x_terms(self):
        return self._count_perp_terms()

    def count_y_terms(self):
        return self._count_perp_terms()

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

        B, dens, temp, mass, charge, alpha, ky, kz = self.vt.make_value_or_expression(
            self)
        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import (jperp_terms,
                                                                   build_perp_coefficients)

        # nmax +1 to use recurrent rules for the bessel functions.
        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin)
        self._jitted_coeffs = build_perp_coefficients(ind_vars, ky, kz, omega, B, dens,
                                                      temp, mass, charge, alpha, fits,
                                                      self.An_mode, self.use_4_components,
                                                      self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_nmax'] = 5
        v['ra_kprmax'] = 15
        v['ra_mmin'] = 7
        v['An_mode'] = "kpara->0"
        v['use_4_components'] = "xx-xy-yx-yy"
        v['debug_option'] = ''
        return v

    def panel1_param(self):
        panels = super(NonlocalJ1D_Jperp, self).panel1_param()
        panels.extend([["An", None, 1, {"values": ["kpara->0", "kpara from kz"]}],
                       ["Components", None, 1, {
                           "values": ["xx only", "xx-xy-yx-yy"]}],
                       ["cyclotron harms.", None, 400, {}],
                       #["-> RA. options", None, None, {"no_tlw_resize": True}],
                       ["max (kp*rho)", None, 300, {}],
                       ["#terms min.", None, 400, {}],
                       ["debug opts.", '', 0, {}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NonlocalJ1D_Jperp, self).get_panel1_value()
        values.extend([self.An_mode, self.use_4_components,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.debug_option])
        return values

    def import_panel1_value(self, v):

        check = super(NonlocalJ1D_Jperp, self).import_panel1_value(v)
        self.An_mode = str(v[-6])
        self.use_4_components = str(v[-5])
        self.ra_nmax = int(v[-4])
        self.ra_kprmax = float(v[-3])
        self.ra_mmin = int(v[-2])
        self.debug_option = str(v[-1])
        return True

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)
        if dep_var in self.get_jx_names():
            return True
        if dep_var in self.get_jy_names():
            return True
        return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        Jnl = Jn1 + Jx2 + Jx3 ..
        nabla^2 Jx1 - d1 = c1 * E
        nabla^2 Jx1 - d2 = c2 * E
        nabla^2 Jx3 - d1 = c3 * E
        '''
        Jnlxterms = self.get_jx_names()
        Jnlyterms = self.get_jy_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exname = var_s[0]
        Eyname = var_s[1]

        loc = []
        for n in Jnlxterms:   # Ex -> Jx
            loc.append((n, Exname, 1, 1))
        for n in Jnlxterms:   # Ey -> Jx
            loc.append((n, Eyname, 1, 1))
        for n in Jnlxterms:   # Jx -> Ex
            loc.append((Exname, n, 1, 1))

        for n in Jnlyterms:    # Ex -> Jy
            loc.append((n, Exname, 1, 1))
        for n in Jnlyterms:    # Ey -> Jy
            loc.append((n, Eyname, 1, 1))
        for n in Jnlyterms:    # Jy -> Ey
            loc.append((Eyname, n, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        if dep_var in jxnames:
            idx = jxnames.index(dep_var)
        if dep_var in jynames:
            idx = jynames.index(dep_var)

        coeffs, _coeff5 = self._jitted_coeffs

        kappa = coeffs["kappa"]
        if idx != 0:
            message = "Add diffusion and mass integrator contribution"
            self.add_integrator(engine, 'diffusion', -kappa, a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            dd = coeffs["dterms"][idx-1]
            self.add_integrator(engine, 'mass', -dd, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        else:  # constant term contribution

            if real:
                message = "Add mass integrator contribution"
                coeff = mfem.ConstantCoefficient(-1)
                self.add_integrator(engine, 'mass', coeff, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
            else:
                message = "No integrator contribution"
        if real:
            dprint1(message, "(real)",  dep_var, idx)
        else:
            dprint1(message, "(imag)",  dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        idx = -1
        jx = False

        if r in jxnames:
            jx = True
            idx = jxnames.index(r)
            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]
        if r in jynames:
            jx = False
            idx = jynames.index(r)
            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

        if real:
            dprint1("Add mixed contribution(real)"  "r/c", r, c, idx, jx)
        else:
            dprint1("Add mixed contribution(imag)"  "r/c", r, c, idx, jx)

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]
        var_s = em1d.dep_vars
        Exname = var_s[0]
        Eyname = var_s[1]

        if c == Exname and jx:
            # Ex -> Jx
            ccoeff = slot["diag"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif c == Exname and not jx:
            # Ex -> Jy
            ccoeff = -slot["xy"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            ccoeff = slot["cross_grad"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarDerivativeIntegrator)
        elif c == Eyname and jx:
            # Ey -> Jx
            ccoeff = slot["xy"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            ccoeff = slot["cross_grad"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarDerivativeIntegrator)
        elif c == Eyname and not jx:
            # Ey -> Jy
            ccoeff = slot["diag"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            ccoeff = slot["diffusion"]
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedGradGradIntegrator)

        elif r == Exname or r == Eyname:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            if not real:  # -j omega
                omega = 2*np.pi*em1d.freq
                sc = mfem.ConstantCoefficient(-omega)
                self.add_integrator(engine, 'jcontribution', sc,
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            if real:  # alpha J
                alpha = self._jitted_coeffs[1]
                self.add_integrator(engine, 'jcontribution', alpha,
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

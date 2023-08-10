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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jperp3')

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
                               no_func=True,
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


class NonlocalJ1D_Jperp3(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jperp3, self).__init__(**kwargs)

    @property
    def is_kyzero(self):
        ky = self.get_ky()
        return ky == 0.0

    def get_ky(self):
        if hasattr(self, '_global_ns'):
            B, dens, temp, mass, charge, alpha, ky, kz = self.vt.make_value_or_expression(
                self)
        else:
            ky = 0
        return ky

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
        ngrid = self.ra_ngrid

        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import jperp_terms

        if (self._nmax_bk != nmax or self._kprmax_bk != kprmax or self._mmin_bk != mmin):
            fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                               ngrid=ngrid)
            self._approx_computed = True
            total = 1 + len(fits[0].c_arr)
            self._nperpterms = total
            self._nmax_bk = nmax
            self._kprmax_bk = kprmax
            self._mmin_bk = mmin
            #self._use_4_components = self.use_4_components

        return int(self._nperpterms)

    def get_jx_names(self):
        xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()
        ky = self.get_ky()

        if self.use_4_components == "xx only":
            if ky == 0:
                return xdiag
            else:
                return xdiag + xgrad

        else:
            if ky == 0:
                return xdiag + xcross
            else:
                return xdiag + xcross + xgrad

    def get_jy_names(self):
        xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()
        ky = self.get_ky()

        if self.use_4_components == "xx only":
            return []
        else:
            return ydiag + ycross + ygrad

    def count_x_terms(self):
        return len(self.get_jx_names())

    def count_y_terms(self):
        return len(self.get_jy_names())

    def count_z_terms(self):
        return 0

    def current_names(self):
        # all possible names without considering run-condition
        basex = self.get_root_phys().extra_vars_basex
        basey = self.get_root_phys().extra_vars_basey

        xdiag = [basex + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]
        ydiag = [basey + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]
        xcross = [basex + self.name() + "c" + str(i+1)
                  for i in range(self._count_perp_terms())]
        ycross = [basey + self.name() + "c" + str(i+1)
                  for i in range(self._count_perp_terms())]
        xgrad = [basex + self.name() + "g" + str(i+1)
                 for i in range(self._count_perp_terms())]
        ygrad = [basey + self.name() + "g" + str(i+1)
                 for i in range(self._count_perp_terms())]

        return xdiag, xcross, xgrad, ydiag, ycross, ygrad

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
        ngrid = self.ra_ngrid

        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import jperp_terms
        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp3 import build_perp3_coefficients

        # nmax +1 to use recurrent rules for the bessel functions.
        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                           ngrid=ngrid)
        self._jitted_coeffs = build_perp3_coefficients(ind_vars, ky, kz, omega, B, dens,
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
        v['ra_mmin'] = 5
        v['ra_ngrid'] = 300
        v['An_mode'] = "kpara->0"
        v['use_4_components'] = "xx-xy-yx-yy"
        v['debug_option'] = ''
        return v

    def plot_approx(self, evt):
        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import plot_terms

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        plot_terms(nmax=nmax, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                   ngrid=ngrid)

    def panel1_param(self):
        panels = super(NonlocalJ1D_Jperp3, self).panel1_param()
        panels.extend([["An", None, 1, {"values": ["kpara->0", "kpara from kz", "kpara from kz (w/o damping)"]}],
                       ["Components", None, 1, {
                           "values": ["xx only", "xx-xy-yx-yy"]}],
                       ["cyclotron harms.", None, 400, {}],
                       ["-> RA. options", None, None, {"no_tlw_resize": True}],
                       ["RA max kp*rho", None, 300, {}],
                       ["RA #terms.", None, 400, {}],
                       ["RA #grid.", None, 400, {}],
                       ["<-"],
                       #                       ["debug opts.", '', 0, {}], ])
                       [None, None, 341, {"label": "Check RA.",
                                          "func": 'plot_approx', "noexpand": True}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NonlocalJ1D_Jperp3, self).get_panel1_value()
        values.extend([self.An_mode, self.use_4_components,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self])

        return values

    def import_panel1_value(self, v):

        check = super(NonlocalJ1D_Jperp3, self).import_panel1_value(v)
        self.An_mode = str(v[-7])
        self.use_4_components = str(v[-6])
        self.ra_nmax = int(v[-5])
        self.ra_kprmax = float(v[-4])
        self.ra_mmin = int(v[-3])
        self.ra_ngrid = int(v[-2])
        #self.debug_option = str(v[-1])
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
        xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exname = var_s[0]
        Eyname = var_s[1]

        loc = []
        for n in xdiag:   # Ex -> Jx
            if n in jxnames:
                loc.append((n, Exname, 1, 1))
        for n in xcross:   # Ey -> Jx
            if n in jxnames:
                loc.append((n, Eyname, 1, 1))
        for n in xdiag + xcross:   # Jx -> Ex
            if n in jxnames:
                loc.append((Exname, n, 1, 1))

        for n in ycross:    # Ex -> Jy
            if n in jynames:
                loc.append((n, Exname, 1, 1))
        for n in ydiag:    # Ey -> Jy
            if n in jynames:
                loc.append((n, Eyname, 1, 1))
        for n in ydiag + ycross:    # Jy -> Ey
            if n in jynames:
                loc.append((Eyname, n, 1, 1))

        if self.is_kyzero:
            for n in xgrad:   # Ey -> Jx (gradient)
                if n in jxnames:
                    loc.append((n, Eyname, 1, 1))
            for n in ygrad:   # Ex -> Jy (gradient)
                if n in jynames:
                    loc.append((n, Exname, 1, 1))
            for n in ygrad:   # Ey -> Jy  (gradient)
                if n in jynames:
                    loc.append((n, Eyname, 1, 1))
            for n in xgrad:   # Jx -> Ex
                if n in jxnames:
                    loc.append((Exname, n, 1, 1))
            for n in ygrad:   # Jy -> Ey
                if n in jynames:
                    loc.append((Eyname, n, 1, 1))

        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()

        for items in (xdiag, xcross, xgrad, ydiag, ycross, ygrad):
            if dep_var in items:
                idx = items.index(dep_var)

        coeffs, _coeff5 = self._jitted_coeffs

        if idx != 0:
            message = "Add diffusion and mass integrator contribution"

            kappa = coeffs["kappa"]
            self.add_integrator(engine, 'diffusion', -kappa, a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            dd = coeffs["dterms"][idx-1]
            self.add_integrator(engine, 'mass', -dd, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        else:  # constant term contribution

            if real:
                message = "Add mass integrator contribution"
                kappa0 = coeffs["kappa0"]
                self.add_integrator(engine, 'mass', -kappa0, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
            else:
                message = "No integrator contribution"
        if real:
            dprint1(message, "(real)", dep_var, idx)
        else:
            dprint1(message, "(imag)", dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        idx = -1
        jx = False

        facp = self._jitted_coeffs[0]["facp"]
        facm = self._jitted_coeffs[0]["facm"]

        xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()
        for items in (xdiag, xcross, xgrad, ydiag, ycross, ygrad):
            if r in items:
                idx = items.index(r)
                break
            if c in items:
                idx = items.index(c)
                break
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

        if c == Exname and r in xdiag:
            # Ex -> Jx
            ccoeff = slot["diag"]*facp
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif c == Exname and r in ycross:
            # Ex -> Jy
            ccoeff = -slot["xy"]*facp
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            #ccoeff = slot["cross_grad"]
            # self.add_integrator(engine, 'cterm', ccoeff,
            #                    mbf.AddDomainIntegrator, mfem.MixedScalarDerivativeIntegrator)
        elif c == Eyname and r in xcross:
            # Ey -> Jx
            ccoeff = slot["xy"]*facp
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            #ccoeff = slot["cross_grad"]
            # self.add_integrator(engine, 'cterm', ccoeff,
            #                    mbf.AddDomainIntegrator, mfem.MixedScalarDerivativeIntegrator)
        elif c == Eyname and r in ydiag:
            # Ey -> Jy
            ccoeff = slot["diag"]*facp
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
        elif c == Eyname and r in ygrad:
            #ccoeff = slot["diffusion"]*facp
            # self.add_integrator(engine, 'cterm', ccoeff,
            #                    mbf.AddDomainIntegrator, mfem.MixedGradGradIntegrator)
            ccoeff = slot["diffusion"]*facm
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarWeakDerivativeIntegrator)

        if r == Exname and c in xdiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            ccoeff = slot["diag"]*facm
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif r == Exname and c in ycross:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            ccoeff = -slot["xy"]*facm
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif r == Eyname and c in xcross:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            ccoeff = slot["xy"]*facm
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif r == Eyname and c in ydiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            ccoeff = slot["diag"]*facm
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
        elif r == Eyname and c in ygrad:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return

            ccoeff = slot["diffusion"]*facm
            # self.add_integrator(engine, 'cterm', ccoeff,
            #                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarDerivativeIntegrator)

        if r == Exname or r == Eyname:
            if real:  # alpha J
                alpha = self._jitted_coeffs[1]
                self.add_integrator(engine, 'jcontribution', alpha,
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

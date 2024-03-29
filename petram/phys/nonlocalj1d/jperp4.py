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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jperp4')

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
        ('tene', VtableElement('tene', type='array',
                               guilabel='collisions (Te, ne)',
                               default="10, 1e17",
                               tip="electron density and temperature for collision")),
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

anbn_options = ("kpara->0 + col.", "kpara from kz")


class NonlocalJ1D_Jperp4(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jperp4, self).__init__(**kwargs)

    @property
    def is_kyzero(self):
        ky = self.get_ky()
        return ky == 0.0

    def get_ky(self):
        if hasattr(self, '_global_ns'):
            B, dens, temp, mass, charge, tene, ky, kz = self.vt.make_value_or_expression(
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
        B, dens, temp, mass, charge, tene, ky, kz = self.vt.make_value_or_expression(
            self)
        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms

        if (self._nmax_bk != nmax or self._kprmax_bk != kprmax or self._mmin_bk != mmin):
            fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                               ngrid=ngrid)
            self._approx_computed = True
            total = 1 + len(fits[0].c_arr)
            self._nperpterms = total
            self._nmax_bk = nmax
            self._kprmax_bk = kprmax
            self._mmin_bk = mmin
            # self._use_4_components = self.use_4_components

        return int(self._nperpterms)

    def get_jx_names(self):
        xdiag, _ydiag = self.current_names()
        return xdiag

    def get_jy_names(self):
        xdiag, ydiag = self.current_names()

        if self.use_4_components == "xx only":
            return []
        else:
            return ydiag
#            return ydiag + ycross + ygrad

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

        xdiag = ([basex + "u" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())] +
                 [basex + "v" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())])
        ydiag = ([basey + "u" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())] +
                 [basey + "v" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())])
        return xdiag, ydiag

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]

        freq, omega = em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, tene, ky, kz = self.vt.make_value_or_expression(
            self)
        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms
        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp4 import build_perp4_coefficients

        # nmax +1 to use recurrent rules for the bessel functions.
        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                           ngrid=ngrid)
        self._jitted_coeffs = build_perp4_coefficients(ind_vars, ky, kz, omega, B, dens,
                                                       temp, mass, charge, tene, fits,
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
        v['ra_pmax'] = 15
        v['An_mode'] = "kpara->0 + col."
        v['use_4_components'] = "xx-xy-yx-yy"
        v['debug_option'] = ''
        return v

    def plot_approx(self, evt):
        from petram.phys.common.nonlocalj_subs import plot_terms

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        pmax = self.ra_pmax

        plot_terms(nmax=nmax, maxkrsqr=kprmax**2, mmin=mmin, mmax=mmin,
                   ngrid=ngrid, pmax=pmax)

    def panel1_param(self):
        panels = super(NonlocalJ1D_Jperp4, self).panel1_param()
        panels.extend([["An", None, 1, {"values": ["kpara->0 + col.", "kpara from kz"]}],
                       ["Components", None, 1, {
                           "values": ["xx only", "xx-xy-yx-yy"]}],
                       ["cyclotron harms.", None, 400, {}],
                       ["-> RA. options", None, None, {"no_tlw_resize": True}],
                       ["RA max kp*rho", None, 300, {}],
                       ["RA #terms.", None, 400, {}],
                       ["RA #grid.", None, 400, {}],
                       ["Plot max.", None, 300, {}],
                       ["<-"],
                       #                       ["debug opts.", '', 0, {}], ])
                       [None, None, 341, {"label": "Check RA.",
                                          "func": 'plot_approx', "noexpand": True}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NonlocalJ1D_Jperp4, self).get_panel1_value()

        if self.An_mode not in anbn_options:
            self.An_mode = anbn_options[0]
        values.extend([self.An_mode, self.use_4_components,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):

        check = super(NonlocalJ1D_Jperp4, self).import_panel1_value(v)
        self.An_mode = str(v[-8])
        self.use_4_components = str(v[-7])
        self.ra_nmax = int(v[-6])
        self.ra_kprmax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])
        # self.debug_option = str(v[-1])
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
        # xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()
        xdiag, ydiag = self.current_names()

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        basex = self.get_root_phys().extra_vars_basex
        basey = self.get_root_phys().extra_vars_basey

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exname = var_s[0]
        Eyname = var_s[1]

        loc = []
        if self.use_4_components == "xx only":
            for n in xdiag:   # Ex, -> Jx
                if n in jxnames:
                    loc.append((n, Exname, 1, 1))
            for n in xdiag:   # Jx -> Ex
                if n in jxnames:
                    loc.append((Exname, n, 1, 1))

        else:
            for n in xdiag:   # Jx
                if n in jxnames:
                    if n.startswith(basex+"u"):
                        loc.append((n, Exname, 1, 1))
                        loc.append((n, Eyname, 1, 1))
                        loc.append((Exname, n, 1, 1))
                    else:
                        loc.append((n, Exname, 1, 1))
                        loc.append((Exname, n, 1, 1))
                        loc.append((Eyname, n, 1, 1))
            for n in ydiag:   # Jy
                if n in jynames:
                    if n.startswith(basey+"u"):
                        loc.append((n, Exname, 1, 1))
                        loc.append((n, Eyname, 1, 1))
                        loc.append((Eyname, n, 1, 1))
                    else:
                        loc.append((n, Eyname, 1, 1))
                        loc.append((Exname, n, 1, 1))
                        loc.append((Eyname, n, 1, 1))

        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        basex = self.get_root_phys().extra_vars_basex
        basey = self.get_root_phys().extra_vars_basey

        # xdiag, xcross, xgrad, ydiag, ycross, ygrad = self.current_names()
        # for items in (xdiag, xcross, xgrad, ydiag, ycross, ygrad)

        xdiag, ydiag = self.current_names()
        for items in (xdiag, ydiag):
            if dep_var in items:
                idx = items.index(dep_var)
                if idx >= len(items)//2:
                    idx = idx - len(items)//2

        coeffs = self._jitted_coeffs

        if idx != 0:
            message = "Add diffusion and mass integrator contribution"
            if real:
                mone = mfem.ConstantCoefficient(-1.0)
                self.add_integrator(engine, 'diffusion', mone, a.AddDomainIntegrator,
                                    mfem.DiffusionIntegrator)
            if dep_var.startswith(basex+"u") or dep_var.startswith(basey+"u"):
                dd = coeffs["dterms"][idx-1]
            else:
                dd = coeffs["dterms"][idx-1].conj()

            self.add_integrator(engine, 'mass', -dd, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        else:  # constant term contribution

            if real:
                message = "Add mass integrator contribution"
                dd0 = coeffs["dd0"]
                self.add_integrator(engine, 'mass', -dd0, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
            else:
                message = "No integrator contribution"
        if real:
            dprint1(message, "(real)", dep_var, idx)
        else:
            dprint1(message, "(imag)", dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):

        basex = self.get_root_phys().extra_vars_basex
        basey = self.get_root_phys().extra_vars_basey

        jxnames = self.get_jx_names()
        jynames = self.get_jy_names()

        idx = -1
        jx = False

        fac = self._jitted_coeffs["fac"]

        xdiag, ydiag = self.current_names()
        for items in (xdiag, ydiag):

            if r in items:
                idx = items.index(r)
                if idx >= len(items)//2:
                    idx = idx - len(items)//2
                break
            if c in items:
                idx = items.index(c)
                if idx >= len(items)//2:
                    idx = idx - len(items)//2
                break
        if idx == 0:
            slot = self._jitted_coeffs["c0"]
        else:
            slot = self._jitted_coeffs["cterms"][idx-1]

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

        #
        #    off-diag rows
        #
        if c == Exname and r in xdiag:
            # Ex -> Jx
            if r.startswith(basex+"u"):
                ccoeff = slot["diag"] + slot["diagi"]
            else:
                ccoeff = 1j*fac
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif c == Eyname and r in xdiag:
            # Ey -> Jx
            if r.startswith(basex+"u"):
                ccoeff = slot["xy"] + slot["xyi"]
            else:
                ccoeff = 1j*fac
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

            if self.is_kyzero:
                return

            if r.startswith(basex+"u"):
                ccoeff = slot["cross_grad"] + slot["cross_gradi"]
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarWeakDerivativeIntegrator)

        elif c == Exname and r in ydiag:
            # Ex -> Jy
            if r.startswith(basey+"u"):
                ccoeff = -slot["xy"] - slot["xyi"]
            else:
                ccoeff = 1j*fac
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

            if self.is_kyzero:
                return

            if r.startswith(basey+"u"):
                ccoeff = slot["cross_grad"] + slot["cross_gradi"]
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarWeakDerivativeIntegrator)

        elif c == Eyname and r in ydiag:
            # Ey -> Jy

            if r.startswith(basey+"u"):
                ccoeff = slot["diag"] + slot["diagi"]
            else:
                ccoeff = 1j*fac
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            # return
            if r.startswith(basey+"u"):
                ccoeff = slot["diffusion"] + slot["diffusioni"]
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedGradGradIntegrator)

        #
        #    off-diag columns
        #
        elif r == Exname and c in xdiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return
            if not c.startswith(basex+"u"):
                ccoeff = (slot["diag"] - slot["diagi"]).conj()
                # ccoeff = (slot["diag"].conj()*fac)   #.conj()
            else:
                ccoeff = (1j*fac).conj()
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

        elif r == Eyname and c in xdiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return

            # Ey -> Jx
            if not c.startswith(basex+"u"):
                ccoeff = (slot["xy"] - slot["xyi"]).conj()
            else:
                ccoeff = (1j*fac).conj()
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

            if self.is_kyzero:
                return

            if not c.startswith(basex+"u"):
                ccoeff = (slot["cross_grad"] - slot["cross_gradi"]).conj()
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarWeakDerivativeIntegrator)

        elif r == Exname and c in ydiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return

            # Ex -> Jy
            if not c.startswith(basey+"u"):
                ccoeff = (-slot["xy"] + slot["xyi"]).conj()
            else:
                ccoeff = (1j*fac).conj()

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

            if self.is_kyzero:
                return

            if not c.startswith(basey+"u"):
                ccoeff = (slot["cross_grad"] - slot["cross_gradi"]).conj()
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarWeakDerivativeIntegrator)

        elif r == Eyname and c in ydiag:
            if self.debug_option == 'skip_iwJ':
                dprint1("!!!!! skipping counting hot current contribution in EM1D")
                return

            # Ey -> Jy
            if not c.startswith(basey+"u"):
                ccoeff = (slot["diag"] - slot["diagi"]).conj()
            else:
                ccoeff = (1j*fac).conj()

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)
            # return
            if not c.startswith(basey+"u"):
                ccoeff = (slot["diffusion"] - slot["diffusioni"]).conj()
            else:
                return

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator, mfem.MixedGradGradIntegrator)

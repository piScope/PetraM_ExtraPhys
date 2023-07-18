'''

compute non-local current correction.

'''
from petram.phys.nonlocalj2d.nonlocalj2d_model import NonlocalJ2D_BaseDomain
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_Jxxyy')

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
                                          tip="additional damping due to non-local current(sigma*Jhot)")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the z direction")),)


class NonlocalJ2D_Jxxyy(NonlocalJ2D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ2D_Jxxyy, self).__init__(**kwargs)

    def _count_perp_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxyterms = 0
            self._nmax_bk = -1
            self._kprmax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)
        B, dens, temp, masse, charge, alpha, kz = self.vt.make_value_or_expression(
            self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import jperp_terms

        if self._nmax_bk != nmax or self._kprmax_bk != kprmax:
            fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2,
                               mmin=mmin, mmax=mmin,
                               ngrid=ngrid)
            self._approx_computed = True
            total = 1 + len(fits[0].c_arr)
            self._nperpterms = total
            self._nmax_bk = nmax
            self._kprmax_bk = kprmax
            self._mmin_bk = mmin

        return int(self._nperpterms)

    def get_jxy_names(self):
        xydiag, pdiag = self.current_names()
        return xydiag

    def get_jp_names(self):
        xydiag, pdiag = self.current_names()
        return pdiag

    def count_xy_terms(self):
        return len(self.get_jxy_names())

    def count_p_terms(self):
        return len(self.get_jp_names())

    def count_z_terms(self):
        return 0

    def current_names(self):
        # all possible names without considering run-condition
        basexy = self.get_root_phys().extra_vars_basexy
        basep = self.get_root_phys().extra_vars_basep

        xydiag = [basexy + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        pdiag = [basep + self.name() + str(i+2)
                 for i in range(self._count_perp_terms()-1)]
        return xydiag, pdiag

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em2d = mfem_physroot[paired_model]

        freq, omega = em2d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, alpha, kz = self.vt.make_value_or_expression(
            self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.nonlocalj1d.nonlocalj1d_subs_perp import jperp_terms
        from petram.phys.nonlocalj2d.nonlocalj2d_subs_xxyy import build_xxyy_coefficients

        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2,
                           mmin=mmin, mmax=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_xxyy_coefficients(ind_vars, kz, omega, B, dens, temp,
                                                      mass, charge,
                                                      alpha, fits,
                                                      self.An_mode,
                                                      self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_nmax'] = 5
        v['ra_kprmax'] = 15
        v['ra_mmin'] = 3
        v['ra_ngrid'] = 300
        v['An_mode'] = "kpara->0"
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
        panels = super(NonlocalJ2D_Jxxyy, self).panel1_param()
        panels.extend([["An", None, 1, {"values": ["kpara->0", "kpara from kz", "kpara from kz (w/o damping)"]}],
                       ["cyclotron harms.", None, 400, {}],
                       ["-> RA. options", None, None, {"no_tlw_resize": True}],
                       ["RA max kp*rho", None, 300, {}],
                       ["RA #terms.", None, 400, {}],
                       ["RA #grid.", None, 400, {}],
                       ["<-"],
                       # ["debug opts.", '', 0, {}], ])
                       [None, None, 341, {"label": "RA.",
                                          "func": 'plot_approx', "noexpand": True}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NonlocalJ2D_Jxxyy, self).get_panel1_value()
        values.extend([self.An_mode,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self])
        return values

    def import_panel1_value(self, v):

        check = super(NonlocalJ2D_Jxxyy, self).import_panel1_value(v)
        self.An_mode = str(v[-6])
        self.ra_nmax = int(v[-5])
        self.ra_kprmax = float(v[-4])
        self.ra_mmin = int(v[-3])
        self.ra_ngrid = int(v[-2])
        #self.debug_option = str(v[-1])
        return True

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        if check == 3:
            return True

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        xydiag, pdiag = self.current_names()

        jxynames = self.get_jxy_names()
        jpnames = self.get_jp_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exyname = var_s[0]

        loc = []
        for n in xydiag:   # Ex -> Jx
            if n in jxynames:
                loc.append((n, Exyname, 1, 1))
                loc.append((Exyname, n, 1, 1))

        for nxy, np in zip(jxynames[1:], jpnames):
            loc.append((nxy, np, 1, 1))
            loc.append((np, nxy, 1, 1))

        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        jxynames = self.get_jxy_names()
        jpnames = self.get_jp_names()

        # jxy[0] -- constant contribution
        # jxy[1:]  and pdiag are pair

        coeffs, _coeff5 = self._jitted_coeffs

        if dep_var in jxynames:
            idx = jxynames.index(dep_var)
            if idx != 0:
                message = "Add curlcurl + mass integrator contribution"
                kappa = coeffs["kappa"]
                self.add_integrator(engine, 'curlcurl', -kappa, a.AddDomainIntegrator,
                                    mfem.CurlCurlIntegrator)
                dd = coeffs["dterms"][idx-1]
                self.add_integrator(engine, 'mass', -dd, a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)

            else:  # constant term contribution
                message = "Add mass integrator contribution"
                kappa0 = coeffs["kappa0"]
                self.add_integrator(engine, 'mass', -kappa0, a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
        elif dep_var in jpnames:
            if real:  # -1
                mone = mfem.ConstantCoefficient(-1.0)
                self.add_integrator(engine, '-1', mone, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
        else:
            assert False, "should not come here"

        if real:
            dprint1(message, "(real)",  dep_var, idx)
        else:
            dprint1(message, "(imag)",  dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):
        jxynames = self.get_jxy_names()
        jpnames = self.get_jp_names()

        facp = self._jitted_coeffs[0]["facp"]
        facm = self._jitted_coeffs[0]["facm"]

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em2d = mfem_physroot[paired_model]
        var_s = em2d.dep_vars
        Exyname = var_s[0]

        if r == Exyname or c == Exyname:
            if real:
                dprint1("Add mixed cterm contribution(real)"  "r/c",
                        r, c, is_trans)
            else:
                dprint1("Add mixed cterm contribution(imag)"  "r/c",
                        r, c, is_trans)

            if r == Exyname:
                idx = jxynames.index(c)
            else:
                idx = jxynames.index(r)
            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

            if c == Exyname:
                ccoeff = slot["diag"]*facp
                self.add_integrator(engine, 'cterm', ccoeff,
                                    mbf.AddDomainIntegrator, mfem.MixedVectorMassIntegrator)
            else:
                ccoeff = slot["diag"]*facm
                self.add_integrator(engine, 'cterm', ccoeff,
                                    mbf.AddDomainIntegrator, mfem.MixedVectorMassIntegrator)
                if real:  # alpha J
                    alpha = self._jitted_coeffs[1]
                    self.add_integrator(engine, 'jcontribution', alpha,
                                        mbf.AddDomainIntegrator, mfem.MixedVectorMassIntegrator)

        else:
            if real:
                dprint1(
                    "Add mixed vector laplacian contribution(real)"  "r/c", r, c, is_trans)

                one = mfem.ConstantCoefficient(1.0)
                mone = mfem.ConstantCoefficient(-1.0)
                if c in jxynames and r in jpnames:
                    # div
                    self.add_integrator(engine, 'div', one,
                                        mbf.AddDomainIntegrator, mfem.MixedVectorWeakDivergenceIntegrator)

                elif r in jxynames and c in jpnames:
                    # grad
                    self.add_integrator(engine, 'grad', one,
                                        mbf.AddDomainIntegrator, mfem.MixedVectorGradientIntegrator)

            else:
                dprint1(
                    "No vector laplacian mixed-contribution(imag)"  "r/c", r, c, is_trans)

'''

This model consider
   J_perp = wp^2/w n^2 exp(-l)In/l E_perp

'''
from petram.mfem_config import use_parallel
from petram.phys.common.nlj_mixins import NLJ_Jhot
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_Jhot')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('B', VtableElement('bext', type='any',
                            guilabel='magnetic field',
                            default="=[0,0,0]",
                            tip="external magnetic field")),
        ('jacB', VtableElement('jacB', type='any',
                               guilabel='jac(B))',
                               default="",
                               tip="jacobian of B")),
        ('hessB', VtableElement('hessB', type='any',
                                guilabel='hess(B)',
                                default="",
                                tip="hessian of B")),
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
        ('kpa', VtableElement('kpa', type='float',
                              guilabel='k-pa for Zn',
                              default=0,
                              tip="k_parallel used to compute absorption")),
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


def domain_constraints():
    return [NLJ1D_Jhot]


component_options = ("mass", "mass + curlcurl")
anbn_options = ("kpara->0 + col.", "kpara from kz")


class NLJ1D_Jhot(NLJ_Jhot):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ1D_Jhot, self).__init__(**kwargs)

    def _count_perp_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxyterms = 0
            self._nmax_bk = -1
            self._kprmax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms

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


    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]

        freq, omega = em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        gui_setting = self.vt.make_value_or_expression(self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms
        from petram.phys.nlj1d.nlj1d_jhot_subs import build_coefficients

        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2,
                           mmin=mmin, mmax=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_coefficients(ind_vars, omega, gui_setting, fits,
                                                 self.An_mode,
                                                 self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        v = super(NLJ1D_Jhot, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_nmax'] = 5
        v['ra_kprmax'] = 15
        v['ra_mmin'] = 3
        v['ra_ngrid'] = 300
        v['ra_pmax'] = 15
        v['An_mode'] = anbn_options[0]
        v['use_4_components'] = component_options[0]
        v['debug_option'] = ''

        v['use_sigma'] = True
        v['use_delta'] = False
        v['use_tau'] = False
        v['use_pi'] = False
        v['use_eta'] = False
        v['use_xi'] = False
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
        panels = super(NLJ1D_Jhot, self).panel1_param()
        panels.extend([
            ["An", None, 1, {"values": anbn_options}],
            ["Hot terms", None, 36, {"col": 6,
                                     "labels": ('Sig.', 'Del.', 'Tau',
                                                'Pi(NI)', 'Eta', 'Xi')}],
            ["cyclotron harms.", None, 400, {}],
            ["-> RA. options", None, None, {"no_tlw_resize": True}],
            ["RA max kp*rho", None, 300, {}],
            ["RA #terms.", None, 400, {}],
            ["RA #grid.", None, 400, {}],
            ["Plot max.", None, 300, {}],
            ["<-"],
            # ["debug opts.", '', 0, {}], ])
            [None, None, 341, {"label": "Check RA.",
                               "func": 'plot_approx', "noexpand": True}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NLJ1D_Jhot, self).get_panel1_value()

        if self.An_mode not in anbn_options:
            self.An_mode = anbn_options[0]

        values.extend([self.An_mode,
                       [self.use_sigma, self.use_delta, self.use_tau,
                        self.use_pi, self.use_eta, self.use_xi],
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):
        check = super(NLJ1D_Jhot, self).import_panel1_value(v)
        self.An_mode = str(v[-8])
        self.use_sigma = bool(v[-7][-6][1])
        self.use_delta = bool(v[-7][-5][1])
        self.use_tau = bool(v[-7][-4][1])
        self.use_pi = bool(v[-7][-3][1])
        self.use_eta = bool(v[-7][-2][1])
        self.use_xi = bool(v[-7][-1][1])
        self.ra_nmax = int(v[-6])
        self.ra_kprmax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])

        #self.debug_option = str(v[-1])
        return True


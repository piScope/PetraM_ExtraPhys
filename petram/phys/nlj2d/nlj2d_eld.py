'''

This model handls imaginary part of zz component for electrons.

'''
from petram.phys.common.nlj_common_eld import eld_options
from petram.mfem_config import use_parallel
from petram.phys.common.nlj_mixins import NLJ_ELD
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ2D_ELD')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('B', VtableElement('bext', type='any',
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
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the z direction")),)


def domain_constraints():
    return [NLJ2D_ELD]


class NLJ2D_ELD(NLJ_ELD):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ2D_ELD, self).__init__(**kwargs)

    def get_itg_params(self):
        return (3, 3), (3, 3, (0, 1, -1))

    def _count_eld_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_mmin_bk"):
            self._nxyterms = 0
            self._zetamax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        eld_option = self.eld_option

        from petram.phys.common.nlj_ra import eld_terms

        if self._mmin_bk != mmin or self._zetamax_bk != zetamax:
            fits = eld_terms(eld_option, maxzeta=zetamax,
                             mmin=mmin, ngrid=ngrid)
            self._approx_computed = True
            total = 1 + len(fits[0].c_arr)
            self._nperpterms = total
            self._zetamax_bk = zetamax
            self._mmin_bk = mmin

        return int(self._nperpterms)

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em2d = mfem_physroot[paired_model]

        freq, omega = em2d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        gui_setting = self.vt.make_value_or_expression(self)

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        eld_option = self.eld_option

        from petram.phys.common.nlj_ra import eld_terms
        from petram.phys.nlj2d.nlj2d_eld_subs import build_coefficients

        fits = eld_terms(eld_option, maxzeta=zetamax, mmin=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_coefficients(ind_vars, omega, gui_setting, fits,
                                                 self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        v = super(NLJ2D_ELD, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_zetamax'] = 0.85
        v['ra_mmin'] = 4
        v['ra_ngrid'] = 800
        v['ra_pmax'] = 15
        v['eld_option'] = eld_options[0]
        v['debug_option'] = ''

        return v

    def plot_approx(self, evt):
        from petram.phys.common.nlj_ra import plot_eld_terms

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        pmax = self.ra_pmax
        eld_option = self.eld_option

        plot_eld_terms(eld_option, maxzeta=zetamax,
                       mmin=mmin, ngrid=ngrid, pmax=pmax)

    def panel1_param(self):
        from petram.pi.panel_txt import txt_zta, txt_sub0

        txt_zta0 = txt_zta + txt_sub0

        panels = super(NLJ2D_ELD, self).panel1_param()
        panels.extend([
            ["Contribution", None, 1, {"values": eld_options}],
            ["-> RA. options", None, None, {"no_tlw_resize": True}],
            ["max 1/"+txt_zta0, None, 300, {}],
            ["#terms.", None, 400, {}],
            ["#grid.", None, 400, {}],
            ["Plot max.", None, 300, {}],
            ["<-"],
            # ["debug opts.", '', 0, {}], ])
            [None, None, 341, {"label": "Check RA.",
                               "func": 'plot_approx', "noexpand": True}], ])
        # ["<-"],])

        return panels

    def get_panel1_value(self):
        values = super(NLJ2D_ELD, self).get_panel1_value()

        if self.eld_option not in eld_options:
            self.eld_option = eld_options[0]

        values.extend([self.eld_option,
                       self.ra_zetamax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):
        check = super(NLJ2D_ELD, self).import_panel1_value(v)
        self.eld_option = v[-6]
        self.ra_zetamax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])

        #self.debug_option = str(v[-1])
        return True

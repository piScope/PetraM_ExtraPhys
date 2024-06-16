'''

This model handls imaginary part of zz component for electrons.

'''
from petram.phys.common.nlj_common_cyclabs import cyclabs_options
from petram.mfem_config import use_parallel
from petram.phys.common.nlj_mixins import NLJ_CYCLABS
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_CYCLABS')

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
        ('kpe', VtableElement('kpa', type='float',
                              guilabel='k-pe for In',
                              default=0,
                              tip="k_perp used to compute In")),
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
    return [NLJ1D_CYCLABS]


class NLJ1D_CYCLABS(NLJ_CYCLABS):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ1D_CYCLABS, self).__init__(**kwargs)

    def get_itg_params(self):
        return (3, 3), (3, 3, (0, -1, -1))

    def _count_cyclabs_terms(self):
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
        cyclabs_option = self.cyclabs_option

        from petram.phys.common.nlj_ra import cyclabs_terms

        if self._mmin_bk != mmin or self._zetamax_bk != zetamax:
            fits = cyclabs_terms(cyclabs_option, maxzeta=zetamax,
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
        em1d = mfem_physroot[paired_model]

        freq, omega = em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        gui_setting = self.vt.make_value_or_expression(self)

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        cyclabs_option = self.cyclabs_option
        nharm = self.cyclabs_nharm

        from petram.phys.common.nlj_ra import cyclabs_terms
        from petram.phys.nlj1d.nlj1d_cyclabs_subs import build_coefficients

        fits = cyclabs_terms(
            cyclabs_option, maxzeta=zetamax, mmin=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_coefficients(ind_vars, omega, nharm, gui_setting, fits,
                                                 cyclabs_option,
                                                 self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        v = super(NLJ1D_CYCLABS, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_zetamax'] = 5
        v['ra_mmin'] = 6
        v['ra_ngrid'] = 800
        v['ra_pmax'] = 15
        v['cyclabs_option'] = cyclabs_options[0]
        v['cyclabs_nharm'] = 1
        v['debug_option'] = ''

        return v

    def plot_approx(self, evt):
        from petram.phys.common.nlj_ra import plot_cyclabs_terms

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        pmax = self.ra_pmax
        cyclabs_option = self.cyclabs_option

        plot_cyclabs_terms(cyclabs_option, maxzeta=zetamax,
                           mmin=mmin, ngrid=ngrid, pmax=pmax)

    def panel1_param(self):
        from petram.pi.panel_txt import txt_zta, txt_sub0

        txt_zta0 = txt_zta + txt_sub0

        panels = super(NLJ1D_CYCLABS, self).panel1_param()
        panels.extend([
            ["Contribution", None, 1, {"values": cyclabs_options}],
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
        values = super(NLJ1D_CYCLABS, self).get_panel1_value()

        if self.cyclabs_option not in cyclabs_options:
            self.cyclabs_option = cyclabs_options[0]

        values.extend([self.cyclabs_option,
                       self.ra_zetamax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):
        check = super(NLJ1D_CYCLABS, self).import_panel1_value(v)
        self.cyclabs_option = v[-6]
        self.ra_zetamax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])

        #self.debug_option = str(v[-1])
        return True

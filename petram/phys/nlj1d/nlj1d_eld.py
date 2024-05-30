'''

This model handls imaginary part of zz component for electrons.

'''
from petram.phys.common.nlj_common import an_options, bn_options
from petram.mfem_config import use_parallel
from petram.phys.common.nlj_mixins import NLJ_ELD
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_ELD')

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
    return [NLJ1D_ELD]


class NLJ1D_ELD(NLJ_ELD):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ1D_ELD, self).__init__(**kwargs)

    def get_itg_params(self):
        return (3, 3), (3, 3, (0, -1, -1))

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

        from petram.phys.common.nlj_ra import eld_terms

        if self._mmin_bk != mmin or self._zetamax_bk != zetamax:
            fits = eld_terms(maxzeta=zetamax, mmin=mmin, ngrid=ngrid)
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

        from petram.phys.common.nlj_ra import eld_terms
        from petram.phys.nlj1d.nlj1d_eld_subs import build_coefficients

        fits = eld_terms(maxzeta=zetamax, mmin=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_coefficients(ind_vars, omega, gui_setting, fits,
                                                 self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        v = super(NLJ1D_ELD, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['ra_zetamax'] = 0.85
        v['ra_mmin'] = 4
        v['ra_ngrid'] = 800
        v['ra_pmax'] = 15
        v['debug_option'] = ''

        return v

    def plot_approx(self, evt):
        from petram.phys.common.nlj_ra import plot_eld_terms

        zetamax = self.ra_zetamax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid
        pmax = self.ra_pmax

        plot_eld_terms(maxzeta=zetamax, mmin=mmin, ngrid=ngrid, pmax=pmax)

    def panel1_param(self):
        from petram.pi.panel_txt import txt_zta, txt_sub0

        txt_zta0 = txt_zta + txt_sub0

        panels = super(NLJ1D_ELD, self).panel1_param()
        panels.extend([
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
        values = super(NLJ1D_ELD, self).get_panel1_value()
        values.extend([self.ra_zetamax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):
        check = super(NLJ1D_ELD, self).import_panel1_value(v)
        self.ra_zetamax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])

        #self.debug_option = str(v[-1])
        return True

    def get_ju_names(self):
        names = self.current_names_xyz()
        return names[0]

    def get_jv_names(self):
        names = self.current_names_xyz()
        return names[1]

    def count_u_terms(self):
        return len(self.get_ju_names())

    def count_v_terms(self):
        return len(self.get_jv_names())

    def current_names_xyz(self):
        # all possible names without considering run-condition
        baseu = self.get_root_phys().extra_vars_baseu
        basev = self.get_root_phys().extra_vars_basev
        udiag = [baseu + self.name() + str(i+1)
                 for i in range(self._count_eld_terms())]
        vdiag = [basev + self.name() + str(i+1)
                 for i in range(self._count_eld_terms())]

        return udiag, vdiag

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)

        dep_var = root.kfes2depvar(kfes)

        names = self.current_names_xyz()
        udiag, vdiag = names
        all_names = udiag + vdiag

        if dep_var not in all_names:
            return False

        if check == 18:     # u-component
            return True
        elif check == 19:   # v-component
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        root = self.get_root_phys()
        dep_vars = root.dep_vars

        names = self.current_names_xyz()
        udiag, vdiag = names

        root = self.get_root_phys()
        i_jt, i_e, _i_pe, _i_pa = self.get_jt_e_pe_pa_idx()
        assert i_jt >= 0 and i_e >= 0, "Jt or E is not found in dependent variables."

        loc = []
        for name in udiag + vdiag:
            loc.append((name, dep_vars[i_e], 1, 1))
            loc.append((dep_vars[i_jt], name, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorDiffusionIntegrator,)

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        idx, umode, flag = self.get_dep_var_idx(dep_var)

        # ju[0], jv[0]    -- constant contribution
        # ju[1:], jv[1:] --- diffusion contribution

        itg2, itg3 = self.get_itg_params()

        if idx != 0:
            message = "Add diffusion + mass integrator contribution"
            mat = self._jitted_coeffs["weak_lap_para"]
            self.add_integrator(engine, 'diffusion', mat, a.AddDomainIntegrator,
                                PyVectorDiffusionIntegrator,
                                itg_params=itg3)

            if umode:
                dterm = self._jitted_coeffs["dterms"][idx-1]
            else:
                dterm = self._jitted_coeffs["dterms"][idx-1].conj()

            dterm = self._jitted_coeffs["eye3x3"]*dterm
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=itg2)

        else:  # constant term contribution
            message = "Add mass integrator contribution"
            dterm = self._jitted_coeffs["eye3x3"]*self._jitted_coeffs["dd0"]
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=itg2)
        if real:
            dprint1(message, "(real)", dep_var, idx)
        else:
            dprint1(message, "(imag)", dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, row, col, is_trans, _is_conj,
                              real=True):
        '''
        fill mixed contribution
        '''
        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorPartialIntegrator,
                                                PyVectorPartialPartialIntegrator)

        from petram.phys.common.nlj_common import an_options, bn_options

        root = self.get_root_phys()
        dep_vars = root.dep_vars

        jomega = self._jitted_coeffs["jomega"]
        mbpara = self._jitted_coeffs["mbpara"]

        if real:
            dprint1("Add mixed cterm contribution(real)"  "r/c",
                    row, col, is_trans)
        else:
            dprint1("Add mixed cterm contribution(imag)"  "r/c",
                    row, col, is_trans)

        i_jt, i_e, _i_pe, _i_pa = self.get_jt_e_pe_pa_idx()
        itg2, itg3 = self.get_itg_params()

        if col == dep_vars[i_e]:   # E -> Ju, Jv
            idx, umode, flag = self.get_dep_var_idx(row)
            cterm = self._jitted_coeffs["cterms"][idx]

            if umode:
                mat2 = mbpara*cterm
                self.add_integrator(engine,
                                    'mat2',
                                    mat2,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)

            else:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega.conj()
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)
            return

        if row == dep_vars[i_jt]:  # Ju, Jv -> Jt
            idx, umode, flag = self.get_dep_var_idx(col)
            cterm = self._jitted_coeffs["cterms"][idx]

            if umode:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)
            else:
                mat2 = mbpara*cterm
                self.add_integrator(engine,
                                    'mat2',
                                    mat2,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)

            return
        dprint1("No mixed-contribution:"  "r/c ", row, col, is_trans)

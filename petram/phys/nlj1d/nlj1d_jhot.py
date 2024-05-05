'''

This model consider
   J_perp = wp^2/w n^2 exp(-l)In/l E_perp

'''
from petram.phys.common.vector_fe_helper import VectorFEHelper_mxin
from petram.mfem_config import use_parallel
from petram.phys.nlj1d.nlj1d_model import NLJ1D_BaseDomain
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

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


class NLJ1D_Jhot(NLJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ1D_Jhot, self).__init__(**kwargs)

    @property
    def need_e(self):
        return True

    def _count_perp_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxyterms = 0
            self._nmax_bk = -1
            self._kprmax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)
        B, dens, temp, masse, charge, tene, kpa, ky, kz = self.vt.make_value_or_expression(
            self)

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
                 for i in range(self._count_perp_terms())]
        vdiag = [basev + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]

        return udiag, vdiag

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em1d = mfem_physroot[paired_model]

        freq, omega = em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, tene, kpa, ky, kz = self.vt.make_value_or_expression(
            self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms
        from petram.phys.nlj1d.nlj1d_jhot_subs import build_coefficients

        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2,
                           mmin=mmin, mmax=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_coefficients(ind_vars, ky, kz, omega, B, dens, temp,
                                                 mass, charge,
                                                 tene, kpa,  fits,
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
            ["use Sigma (hot S)", False, 3, {"text": ' '}],
            ["use Delta (hot D)", False, 3, {"text": ' '}],
            ["use Tau (hot Syy)", False, 3, {"text": ' '}],
            ["use Pi (NI)", False, 3, {"text": ' '}],
            ["use Eta (hot XZ)", False, 3, {"text": ' '}],
            ["use Xi (NI)", False, 3, {"text": ' '}],
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

        values.extend([self.An_mode, self.use_sigma, self.use_delta, self.use_tau,
                       self.use_pi, self.use_eta, self.use_xi,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])

        return values

    def import_panel1_value(self, v):
        check = super(NLJ1D_Jhot, self).import_panel1_value(v)
        self.An_mode = str(v[-13])
        self.use_sigma = bool(v[-12])
        self.use_delta = bool(v[-11])
        self.use_tau = bool(v[-10])
        self.use_pi = bool(v[-9])
        self.use_eta = bool(v[-8])
        self.use_xi = bool(v[-7])
        self.ra_nmax = int(v[-6])
        self.ra_kprmax = float(v[-5])
        self.ra_mmin = int(v[-4])
        self.ra_ngrid = int(v[-3])
        self.ra_pmax = float(v[-2])

        #self.debug_option = str(v[-1])
        return True

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

    def _get_dep_var_idx(self, dep_var):
        names = self.current_names_xyz()
        udiag, vdiag = names
        if dep_var in udiag:
            idx = udiag.index(dep_var)
            umode = True
            flag = 18
        elif dep_var in vdiag:
            idx = vdiag.index(dep_var)
            umode = False
            flag = 19
        else:
            assert False, "should not come here" + str(dep_var)
        return idx, umode, flag

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorWeakPartialPartialIntegrator,)

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        idx, umode, flag = self._get_dep_var_idx(dep_var)

        # ju[0], jv[0]    -- constant contribution
        # ju[1:], jv[1:] --- diffusion contribution

        if idx != 0:
            message = "Add diffusion + mass integrator contribution"
            mat = self._jitted_coeffs["weak_lap_perp"]
            self.add_integrator(engine, 'diffusion', mat, a.AddDomainIntegrator,
                                PyVectorWeakPartialPartialIntegrator,
                                itg_params=(3, 3, (0, -1, -1)))

            if umode:
                dterm = self._jitted_coeffs["dterms"][idx-1]
            else:
                dterm = self._jitted_coeffs["dterms"][idx-1].conj()

            dterm = self._jitted_coeffs["eye3x3"]*dterm
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=(3, 3, ))

        else:  # constant term contribution
            message = "Add mass integrator contribution"
            dterm = self._jitted_coeffs["eye3x3"]*self._jitted_coeffs["dd0"]
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=(3, 3, ))
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

        root = self.get_root_phys()
        dep_vars = root.dep_vars

        meye = self._jitted_coeffs["meye3x3"]
        mbcross = self._jitted_coeffs["mbcross"]
        mbcrosst = self._jitted_coeffs["mbcrosst"]
        jomega = self._jitted_coeffs["jomega"]
        mbperp = self._jitted_coeffs["mbperp"]

        if real:
            dprint1("Add mixed cterm contribution(real)"  "r/c",
                    row, col, is_trans)
        else:
            dprint1("Add mixed cterm contribution(imag)"  "r/c",
                    row, col, is_trans)

        i_jt, i_e, _i_pe, _i_pa = self.get_jt_e_pe_pa_idx()

        if col == dep_vars[i_e]:   # E -> Ju, Jv
            idx, umode, flag = self._get_dep_var_idx(row)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                if self.use_sigma:
                    ccoeff = mbperp*slot["diag+diagi"]
                    self.add_integrator(engine,
                                        'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=(3, 3, ),)

                if self.use_delta:
                    ccoeff = mbcross*slot["xy+xyi"]
                    self.add_integrator(engine,
                                        'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=(3, 3, ),)
                if self.use_tau:
                    mat = self._jitted_coeffs["mcurlpecurlpe"]*slot["cl+cli"]
                    self.add_integrator(engine,
                                        'diffusion',
                                        mat,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialPartialIntegrator,
                                        itg_params=(3, 3, (0, -1, -1)))

                if self.use_eta:
                    mat = self._jitted_coeffs["mbxcurlpe"] * \
                        slot["eta+etai"]*(-1)
                    self.add_integrator(engine,
                                        'eta',
                                        mat,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=(3, 3, (0, -1, -1)))

                #ccoeff = slot["(diag1+diagi1)*Mpara"]
                # self.fill_divgrad_matrix(
                #    engine, mbf, rowi, colj, ccoeff, real, kz=kz)
            else:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega.conj()
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=(3, 3, ),)

            return
        '''
        if col == dep_vars[i_pa]:   # Epara -> Jm, Jn
            idx, umode, flag = self._get_dep_var_idx(row)
            assert flag in (
                24, 25), "Epara should contribute only on Jm and Jn"

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                if self.use_eta:
                    mat = self._jitted_coeffs["mbxcurlpe"]*slot["eta+etai"]
                    self.add_integrator(engine,
                                        'eta',
                                        mat,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=(3, 3, (0, -1, -1)))

            else:
                ccoeff = jomega.conj()
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=(3, 3, ),)
        '''
        if row == dep_vars[i_jt]:  # Ju, Jv -> Jt
            idx, umode, flag = self._get_dep_var_idx(col)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=(3, 3, ),)

            else:
                if flag in (18, 19):  # Ju, Jv
                    if self.use_sigma:
                        ccoeff = mbperp*slot["conj(diag-diagi)"]
                        self.add_integrator(engine, 'mass',
                                            ccoeff,
                                            mbf.AddDomainIntegrator,
                                            PyVectorMassIntegrator,
                                            itg_params=(3, 3, ),)

                    if self.use_delta:
                        ccoeff = mbcrosst*slot["conj(xy-xyi)"]
                        self.add_integrator(engine,
                                            'mass',
                                            ccoeff,
                                            mbf.AddDomainIntegrator,
                                            PyVectorMassIntegrator,
                                            itg_params=(3, 3, ),)

                    if self.use_tau:
                        mat = self._jitted_coeffs["mcurlpecurlpet"] * \
                            slot["conj(cl-cli)"]
                        self.add_integrator(engine,
                                            'diffusion',
                                            mat,
                                            mbf.AddDomainIntegrator,
                                            PyVectorPartialPartialIntegrator,
                                            itg_params=(3, 3, (0, -1, -1)))

                    if self.use_eta:
                        mat = self._jitted_coeffs["mbxcurlpet"] * \
                            slot["conj(eta-etai)"]*(-1)
                        self.add_integrator(engine,
                                            'eta',
                                            mat,
                                            mbf.AddDomainIntegrator,
                                            PyVectorPartialIntegrator,
                                            itg_params=(3, 3, (0, -1, -1)))
                  #ccoeff = slot["conj(diag1-diagi1)*Mpara"]
                  # self.fill_divgrad_matrix(
                  #    engine, mbf, rowi, colj, ccoeff, real, kz=kz)
                else:
                    assert False, "should not come here"
                '''
                elif flag in (24, 25):  # Jm, Jn
                    if self.use_eta:
                        mat = self._jitted_coeffs["mbxcurlpet"] * \
                            slot["conj(eta-etai)"]
                        self.add_integrator(engine,
                                            'eta',
                                            mat,
                                            mbf.AddDomainIntegrator,
                                            PyVectorPartialIntegrator,
                                            itg_params=(3, 3, (0, -1, -1)))
                '''

            return

        dprint1("No mixed-contribution"  "r/c", row, col, is_trans)

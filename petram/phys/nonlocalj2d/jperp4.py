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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_Jxxyy4')

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
                               no_func=True,
                               tip="mass. normalized by Da. For electrons, use q_Da")),
        ('charge_q', VtableElement('charge_q', type='float',
                                   guilabel='charges(/q)',
                                   default="1",
                                   no_func=True,
                                   tip="charges normalized by q(=1.60217662e-19 [C])")),
        ('tene', VtableElement('tene', type='array',
                               guilabel='collisions (Te, ne)',
                               default="10, 1e17",
                               tip="electron density and temperature for collision")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the z direction")),)


component_options = ("mass", "mass + curlcurl")
anbn_options = ("kpara->0 + col.", "kpara from kz")


class NonlocalJ2D_Jperp4(NonlocalJ2D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ2D_Jperp4, self).__init__(**kwargs)

    def _count_perp_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxyterms = 0
            self._nmax_bk = -1
            self._kprmax_bk = -1.0
            self._mmin_bk = -1

        self.vt.preprocess_params(self)
        B, dens, temp, masse, charge, tene, kz = self.vt.make_value_or_expression(
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

    def get_jxy_names(self):
        names = self.current_names_xyp()
        # xyudiag, xyvdiag, pudiag, pvdiag,
        #   xyrudiag, xyrvdiag, prudiag, prvdiag = names
        return names[0] + names[1] + names[4] + names[5]

    def get_jp_names(self):
        names = self.current_names_xyp()
        # xyudiag, xyvdiag, pudiag, pvdiag,
        #   xyrudiag, xyrvdiag, prudiag, prvdiag = names
        return names[2] + names[3] + names[6] + names[7]

    def count_xy_terms(self):
        return len(self.get_jxy_names())

    def count_p_terms(self):
        return len(self.get_jp_names())

    def count_z_terms(self):
        return 0

    def current_names_xyp(self):
        # all possible names without considering run-condition
        basexy = self.get_root_phys().extra_vars_basexy
        basep = self.get_root_phys().extra_vars_basep

        # xyu, xyv, pu, pv for terms using MassIntegrator
        # xyru, xyrv, pru, prv for terms using CurlCurlIntegrator

        xyudiag = [basexy + "u" + self.name() + str(i+1)
                   for i in range(self._count_perp_terms())]
        xyvdiag = [basexy + "v" + self.name() + str(i+1)
                   for i in range(self._count_perp_terms())]
        pudiag = [basep + "u" + self.name() + str(i+2)
                  for i in range(self._count_perp_terms()-1)]
        pvdiag = [basep + "v" + self.name() + str(i+2)
                  for i in range(self._count_perp_terms()-1)]

        if self.use_4_components == component_options[1]:
            xyrudiag = [basexy + "ru" + self.name() + str(i+1)
                        for i in range(self._count_perp_terms()+1)]
            xyrvdiag = [basexy + "rv" + self.name() + str(i+1)
                        for i in range(self._count_perp_terms()+1)]
            prudiag = [basep + "ru" + self.name() + str(i+2)
                       for i in range(self._count_perp_terms()-1)]
            prvdiag = [basep + "rv" + self.name() + str(i+2)
                       for i in range(self._count_perp_terms()-1)]
        else:
            xyrudiag = []
            xyrvdiag = []
            prudiag = []
            prvdiag = []

        return xyudiag, xyvdiag, pudiag, pvdiag, xyrudiag, xyrvdiag, prudiag, prvdiag

    def current_names(self):
        names = self.current_names_xyp()
        return names

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em2d = mfem_physroot[paired_model]

        freq, omega = em2d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, tene, kz = self.vt.make_value_or_expression(
            self)

        nmax = self.ra_nmax
        kprmax = self.ra_kprmax
        mmin = self.ra_mmin
        ngrid = self.ra_ngrid

        from petram.phys.common.nonlocalj_subs import jperp_terms
        from petram.phys.nonlocalj2d.subs_perp3 import build_perp_coefficients

        fits = jperp_terms(nmax=nmax+1, maxkrsqr=kprmax**2,
                           mmin=mmin, mmax=mmin, ngrid=ngrid)

        self._jitted_coeffs = build_perp_coefficients(ind_vars, kz, omega, B, dens, temp,
                                                      mass, charge,
                                                      tene, fits,
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
        panels = super(NonlocalJ2D_Jperp4, self).panel1_param()
        panels.extend([
            ["An", None, 1, {"values": anbn_options}],
            ["Components", None, 1, {
                "values": component_options}],
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
        values = super(NonlocalJ2D_Jperp4, self).get_panel1_value()

        if self.An_mode not in anbn_options:
            self.An_mode = anbn_options[0]
        if self.use_4_components not in component_options:
            self.use_4_components = component_options[0]

        values.extend([self.An_mode, self.use_4_components,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])
        return values

    def import_panel1_value(self, v):
        check = super(NonlocalJ2D_Jperp4, self).import_panel1_value(v)
        self.An_mode = str(v[-8])
        self.use_4_components = str(v[-7])
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
        if check == 3:     # jxy
            return True
        elif check == 4:   # jp
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        xyudiag, xyvdiag, pudiag, pvdiag, xyrudiag, xyrvdiag, prudiag, prvdiag = self.current_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exyname = var_s[0]
        Ezname = var_s[1]

        loc = []
        for n in xyudiag + xyvdiag:   # Ex -> Jx
            loc.append((n, Exyname, 1, 1))
            loc.append((n, Ezname, 1, 1))
            loc.append((Exyname, n, 1, 1))
            loc.append((Ezname, n, 1, 1))

        for nxy, np in zip(xyudiag[1:], pudiag):
            loc.append((nxy, np, 1, 1))
            loc.append((np, nxy, 1, 1))

        for nxy, np in zip(xyvdiag[1:], pvdiag):
            loc.append((nxy, np, 1, 1))
            loc.append((np, nxy, 1, 1))

        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        xyudiag, xyvdiag, pudiag, pvdiag, xyrudiag, xyrvdiag, prudiag, prvdiag = self.current_names()
        basexy = self.get_root_phys().extra_vars_basexy

        # jxy[0] -- constant contribution
        # jxy[1:]  and pdiag are pair
        if dep_var in xyudiag:
            idx = xyudiag.index(dep_var)
        elif dep_var in xyvdiag:
            idx = xyvdiag.index(dep_var)
        else:
            idx = 0  # not used

        if dep_var in xyudiag + xyvdiag + xyrudiag[:-1] + xyrvdiag[:-1]:
            if idx != 0:
                message = "Add curlcurl or divdiv + mass integrator contribution"
                if real:
                    mone = mfem.ConstantCoefficient(-1.0)
                    self.add_integrator(engine, 'curlcurl', mone, a.AddDomainIntegrator,
                                        mfem.CurlCurlIntegrator)

                if dep_var.startswith(basexy+"u"):
                    dd = self._jitted_coeffs["dterms"][idx-1]
                else:
                    dd = self._jitted_coeffs["dterms"][idx-1].conj()
                self.add_integrator(engine, 'mass', dd, a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)

            else:  # constant term contribution
                message = "Add mass integrator contribution"
                dd0 = self._jitted_coeffs["dd0"]
                self.add_integrator(engine, 'mass', -dd0, a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)

        elif dep_var in xyrudiag[-1:] + xyrvdiag[-1:]:
            message = "Add mass integrator contribution for RE and RJ"
            if real:  # 1
                one = mfem.ConstantCoefficient(1.0)
                self.add_integrator(engine, '1', one, a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
        elif dep_var in pudiag+pvdiag+prudiag+prvdiag:
            message = "Add mass integrator contribution (jp)"
            if real:  # 1
                one = mfem.ConstantCoefficient(1.0)
                self.add_integrator(engine, '1', one, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
        else:
            assert False, "should not come here:" + str(dep_var)

        if real:
            dprint1(message, "(real)",  dep_var, idx)
        else:
            dprint1(message, "(imag)",  dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):

        basexy = self.get_root_phys().extra_vars_basexy
        basep = self.get_root_phys().extra_vars_basep

        xyudiag, xyvdiag, pudiag, pvdiag, xyrudiag, xyrvdiag, prudiag, prvdiag = self.current_names()

        fac = self._jitted_coeffs["fac"]
        U = self._jitted_coeffs["U"]
        Ut = self._jitted_coeffs["Ut"]
        R1 = self._jitted_coeffs["R1"]
        R2 = self._jitted_coeffs["R2"]

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em2d = mfem_physroot[paired_model]
        var_s = em2d.dep_vars
        Exyname = var_s[0]
        Ezname = var_s[1]

        if real:
            dprint1("Add mixed cterm contribution(real)"  "r/c",
                    r, c, is_trans)
        else:
            dprint1("Add mixed cterm contribution(imag)"  "r/c",
                    r, c, is_trans)

        if c == Exyname and r in xyudiag+xyvdiag:
            if r in xyudiag:
                idx = xyudiag.index(r)
            else:
                idx = xyvdiag.index(r)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            u_22 = U[[0, 1], [0, 1]]

            if r.startswith(basexy+"u"):
                ccoeff_d = slot["diag"] + slot["diagi"]
                ccoeff_c = slot["xy"] + slot["xyi"]

                ccoeff2 = u_22*ccoeff_d + R2.dot(u_22)*ccoeff_c
            else:
                ccoeff = 1j*fac
                ccoeff2 = u_22*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r == Exyname and c in xyudiag+xyvdiag:
            if c in xyudiag:
                idx = xyudiag.index(c)
            else:
                idx = xyvdiag.index(c)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            ut_22 = Ut[[0, 1], [0, 1]]

            if c.startswith(basexy+"v"):
                ccoeff_d = (slot["diag"] - slot["diagi"]).conj()
                ccoeff_c = (slot["xy"] - slot["xyi"]).conj()

                ccoeff2 = ut_22*ccoeff_d - ut_22.dot(R2)*ccoeff_c  # r2^t = -r2

            else:
                ccoeff = (1j*fac).conj()
                ccoeff2 = ut_22*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif c == Ezname and r in xyudiag+xyvdiag:
            if r in xyudiag:
                idx = xyudiag.index(r)
            else:
                idx = xyvdiag.index(r)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            u_12 = U[[0, 1], 2]

            if r.startswith(basexy+"u"):
                ccoeff_d = slot["diag"] + slot["diagi"]
                ccoeff_c = slot["xy"] + slot["xyi"]
                ccoeff2 = u_12*ccoeff_d + R2.dot(u_12)*ccoeff_c
            else:
                ccoeff = 1j*fac
                ccoeff2 = u_12*ccoeff

            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorProductIntegrator)

        elif r == Ezname and c in xyudiag+xyvdiag:
            if c in xyudiag:
                idx = xyudiag.index(c)
            else:
                idx = xyvdiag.index(c)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            ut_21 = Ut[2, [0, 1]]

            if c.startswith(basexy+"v"):
                ccoeff_d = (slot["diag"] - slot["diagi"]).conj()
                ccoeff_c = (slot["xy"] - slot["xyi"]).conj()
                ccoeff2 = ut_21*ccoeff_d + R2.dot(ut_21)*ccoeff_c
            else:
                ccoeff = (1j*fac).conj()
                ccoeff2 = ut_21*ccoeff

            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)

        elif c == Exyname and r in xyrudiag[-1:]:
            u_22 = U[[0, 1], [0, 1]]
            ccoeff2 = -R1.dot(u_22)
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif c == Ezname and r in xyrudiag[-1:]:
            u_12 = U[[0, 1], 2]
            ccoeff2 = -R1.dot(u_12)
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r == Exyname and c in xyrvdiag[-1:]:
            ccoeff = (1j*fac).conj()
            ut_22 = Ut[[0, 1], [0, 1]]
            ccoeff2 = R1.dot(ut_22)*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r == Ezname and c in xyrvdiag[-1:]:
            ccoeff = (1j*fac).conj()
            ut_12 = Ut[2, [0, 1]]
            ccoeff2 = R1.dot(ut_12)*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif c == Exyname and r in xyrudiag[:-1]:
            ccoeff = 1j*fac
            u_22 = Ut[[0, 1], [0, 1]]
            ccoeff2 = u_22.dot(R1).ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif c == Ezname and r in xyrudiag[:-1]:
            ccoeff = 1j*fac
            u_12 = U[[0, 1], 2]
            ccoeff2 = R1.dot(u_12).ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r == Exyname and c == xyrudiag[-1]:
            ut_22 = Ut[[0, 1], [0, 1]]
            ccoeff2 = -ut_22.dot(R1)
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r == Ezname and c == xyrudiag[-1]:
            ut_12 = Ut[2, [0, 1]]
            ccoeff2 = -ut_12.dot(R1)
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorMassIntegrator)

        elif r in xyrudiag[:-1] and c == xyrudiag[-1]:
            #curl-curl (Lp)
            idx = xyrudiag.index(r)
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            ccoeff = -(slot["rcurlcurlr"] + slot["rcurlcurli"])
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedCurlCurlIntegrator)

        elif r == xyrvdiag[-1] and c in xyrvdiag[:-1]:
            #curl-curl (Lm)
            idx = xyrvdiag.index(c)
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            ccoeff = (slot["rcurlcurli"] - slot["rcurlcurlr"]).conj()
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedCurlCurlIntegrator)

        else:
            if real:
                dprint1(
                    "Add mixed vector laplacian contribution(real)"  "r/c", r, c, is_trans)

                if c in xyudiag + xyvdiag and r in pudiag + pvdiag:
                    # div
                    mone = mfem.ConstantCoefficient(-1.0)
                    self.add_integrator(engine, 'div', mone,
                                        mbf.AddDomainIntegrator,
                                        mfem.MixedVectorWeakDivergenceIntegrator)
                elif c in xyrudiag[:-1] + xyrvdiag[:-1] and r in prudiag + prvdiag:
                    # div
                    mone = mfem.ConstantCoefficient(-1.0)
                    self.add_integrator(engine, 'div', mone,
                                        mbf.AddDomainIntegrator,
                                        mfem.MixedVectorWeakDivergenceIntegrator)

                elif r in xyudiag + xyvdiag and c in pudiag + pvdiag:
                    # grad
                    one = mfem.ConstantCoefficient(1.0)
                    self.add_integrator(engine, 'grad', one,
                                        mbf.AddDomainIntegrator,
                                        mfem.MixedVectorGradientIntegrator)
                elif r in xyrudiag[:-1] + xyrvdiag[:-1] and c in prudiag + prvdiag:
                    # grad
                    one = mfem.ConstantCoefficient(1.0)
                    self.add_integrator(engine, 'grad', one,
                                        mbf.AddDomainIntegrator,
                                        mfem.MixedVectorGradientIntegrator)
            else:
                dprint1(
                    "No vector laplacian mixed-contribution(imag)"  "r/c", r, c, is_trans)

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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_Jxxyy3')

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


class NonlocalJ2D_Jxxyy3(NonlocalJ2D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ2D_Jxxyy3, self).__init__(**kwargs)

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
        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names_xyp()
        return xyudiag + xyvdiag

    def get_jp_names(self):
        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names_xyp()
        return pudiag + pvdiag

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

        xyudiag = [basexy + "u" + self.name() + str(i+1)
                   for i in range(self._count_perp_terms())]
        xyvdiag = [basexy + "v" + self.name() + str(i+1)
                   for i in range(self._count_perp_terms())]
        pudiag = [basep + "u" + self.name() + str(i+2)
                  for i in range(self._count_perp_terms()-1)]
        pvdiag = [basep + "v" + self.name() + str(i+2)
                  for i in range(self._count_perp_terms()-1)]

        return xyudiag, xyvdiag, pudiag, pvdiag

    def current_names(self):
        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names_xyp()
        return xyudiag, xyvdiag, pudiag, pvdiag

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

        self._jitted_coeffs = build_xxyy_coefficients(ind_vars, kz, omega, B, dens, temp,
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
        v['An_mode'] = "kpara->0"
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
        panels = super(NonlocalJ2D_Jxxyy3, self).panel1_param()
        panels.extend([
            ["An", None, 1, {"values": [
                "kpara->0", "kpara from kz", "kpara from kz (w/o damping)"]}],
            ["Components", None, 1, {
                "values": ["xx only", "xx-xy-yx-yy"]}],
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
        values = super(NonlocalJ2D_Jxxyy3, self).get_panel1_value()
        values.extend([self.An_mode, self.use_4_components,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])
        return values

    def import_panel1_value(self, v):
        check = super(NonlocalJ2D_Jxxyy3, self).import_panel1_value(v)
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
        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names()

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

        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names()

        # jxy[0] -- constant contribution
        # jxy[1:]  and pdiag are pair

        coeffs, _coeff5 = self._jitted_coeffs

        if dep_var in xyudiag + xyvdiag:
            if dep_var in xyudiag:
                idx = xyudiag.index(dep_var)
            else:
                idx = xyvdiag.index(dep_var)

            if idx != 0:
                message = "Add curlcurl or divdiv + mass integrator contribution"
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

        elif dep_var in pudiag+pvdiag:
            message = "Add mass integrator contribution (jp)"
            if real:  # -1
                mone = mfem.ConstantCoefficient(-1.0)
                self.add_integrator(engine, '-1', mone, a.AddDomainIntegrator,
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

        xyudiag, xyvdiag, pudiag, pvdiag = self.current_names()

        fac = self._jitted_coeffs[0]["fac"]
        U = self._jitted_coeffs[0]["U"]
        Ut = self._jitted_coeffs[0]["Ut"]

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
            if dep_var in xyudiag:
                idx = xyudiag.index(r)
            else:
                idx = xyvdiag.index(r)

            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

            u_22 = U[[0, 1], [0, 1]]

            if r.startswith(basex+"u"):
                ccoeff = slot["diag"] + slot["diagi"]
            else:
                ccoeff = 1j*fac

            ccoeff2 = u_22*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,  mfem.MixedVectorMassIntegrator)

        elif r == Exyname and c in xyudiag+xyvdiag:
            if dep_var in xyudiag:
                idx = xyudiag.index(c)
            else:
                idx = xyvdiag.index(c)

            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

            ut_22 = Ut[[0, 1], [0, 1]]

            if c.startswith(basex+"v"):
                ccoeff = (slot["diag"] - slot["diagi"]).conj()
            else:
                ccoeff = 1j*fac

            ccoeff2 = ut_22*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator, mfem.MixedVectorMassIntegrator)

        elif c == Ezname and r in xyudiag+xyvdiag:
            if dep_var in xyudiag:
                idx = xyudiag.index(r)
            else:
                idx = xyvdiag.index(r)

            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

            if r.startswith(basex+"u"):
                ccoeff = slot["diag"] + slot["diagi"]
            else:
                ccoeff = 1j*fac

            u_12 = U[[0, 1], 2]
            ccoeff2 = u_12*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,  mfem.MixedVectorProductIntegrator)

        elif r == Ezname and c in xyudiag+xyvdiag:
            if dep_var in xyudiag:
                idx = xyudiag.index(c)
            else:
                idx = xyvdiag.index(c)

            if idx == 0:
                slot = self._jitted_coeffs[0]["c0"]
            else:
                slot = self._jitted_coeffs[0]["cterms"][idx-1]

            if c.startswith(basex+"v"):
                ccoeff = (slot["diag"] - slot["diagi"]).conj()
            else:
                ccoeff = 1j*fac

            ut_21 = Ut[2, [0, 1]]
            ccoeff2 = u_12*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator, mfem.MixedDotProductIntegrator)

        else:
            if real:
                dprint1(
                    "Add mixed vector laplacian contribution(real)"  "r/c", r, c, is_trans)

                if c in jxynames and r in jpnames:
                    # div
                    one = mfem.ConstantCoefficient(1.0)
                    self.add_integrator(engine, 'div', one,
                                        mbf.AddDomainIntegrator, mfem.MixedVectorWeakDivergenceIntegrator)

                elif r in jxynames and c in jpnames:
                    # grad
                    one = mfem.ConstantCoefficient(1.0)
                    self.add_integrator(engine, 'grad', one,
                                        mbf.AddDomainIntegrator, mfem.MixedVectorGradientIntegrator)
            else:
                dprint1(
                    "No vector laplacian mixed-contribution(imag)"  "r/c", r, c, is_trans)

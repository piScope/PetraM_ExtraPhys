'''

This model consider
   Sxx component
   Syy component is assume to be the same as Sxx

   Jx. Jy Jz are H1 element

   J_perp = wp^2/w n^2 exp(-l)In/l E_perp



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


class NonlocalJ2D_Jxx3(NonlocalJ2D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ2D_Jxx3, self).__init__(**kwargs)

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

    def get_jx_names(self):
        names = self.current_names_xyz()
        return names[0] + names[1]

    def get_jy_names(self):
        names = self.current_names_xyz()
        return names[2] + names[3]

    def get_jz_names(self):
        names = self.current_names_xyz()
        return names[4] + names[5]

    def count_x_terms(self):
        return len(self.get_jx_names())

    def count_y_terms(self):
        return len(self.get_jy_names())

    def count_z_terms(self):
        return len(self.get_jz_names())

    def current_names_xyz(self):
        # all possible names without considering run-condition
        basex = self.get_root_phys().extra_vars_basex
        basey = self.get_root_phys().extra_vars_basey
        basez = self.get_root_phys().extra_vars_basez

        xudiag = [basex + "u" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        xvdiag = [basex + "v" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        yudiag = [basey + "u" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        yvdiag = [basey + "v" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        zudiag = [basez + "u" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]
        zvdiag = [basez + "v" + self.name() + str(i+1)
                  for i in range(self._count_perp_terms())]

        return xudiag, xvdiag, yudiag, yvdiag, zudiag, zvdiag

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
        panels = super(NonlocalJ2D_Jxx3, self).panel1_param()
        panels.extend([
            ["An", None, 1, {"values": anbn_options}],
            # ["Components", None, 1, {
            #    "values": component_options}],
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
        values = super(NonlocalJ2D_Jxx3, self).get_panel1_value()

        if self.An_mode not in anbn_options:
            self.An_mode = anbn_options[0]

        values.extend([self.An_mode,
                       self.ra_nmax, self.ra_kprmax, self.ra_mmin,
                       self.ra_ngrid, self.ra_pmax, self])
        return values

    def import_panel1_value(self, v):
        check = super(NonlocalJ2D_Jxx3, self).import_panel1_value(v)
        self.An_mode = str(v[-7])
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
        if check == 6:     # jx
            return True
        elif check == 7:   # jy
            return True
        elif check == 5:   # jz
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        names = self.current_names_xyz()
        xudiag, xvdiag, yudiag, yvdiag, zudiag, zvdiag = names
        all_names = xudiag + xvdiag + yudiag + yvdiag + zudiag + zvdiag

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Exyname = var_s[0]
        Ezname = var_s[1]

        loc = []
        for n in all_names:
            loc.append((n, Exyname, 1, 1))
            loc.append((n, Ezname, 1, 1))
            loc.append((Exyname, n, 1, 1))
            loc.append((Ezname, n, 1, 1))

        return loc

    def _get_dep_var_idx(self, dep_var, names):
        xudiag, xvdiag, yudiag, yvdiag, zudiag, zvdiag = names
        if dep_var in xudiag:
            idx = xudiag.index(dep_var)
            umode = True
            dirc = 'x'
        elif dep_var in xvdiag:
            idx = xvdiag.index(dep_var)
            umode = False
            dirc = 'x'
        elif dep_var in yudiag:
            idx = yudiag.index(dep_var)
            umode = True
            dirc = 'y'
        elif dep_var in yvdiag:
            idx = yvdiag.index(dep_var)
            umode = False
            dirc = 'y'
        elif dep_var in zudiag:
            idx = zudiag.index(dep_var)
            umode = True
            dirc = 'z'
        elif dep_var in zvdiag:
            idx = zvdiag.index(dep_var)
            umode = False
            dirc = 'z'
        else:
            assert False, "should not come here" + str(dep_var)
        return idx, umode, dirc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        names = self.current_names_xyz()
        idx, umode, _dirc = self._get_dep_var_idx(dep_var, names)

        # jxyz[0] -- constant contribution
        # jxyz[1:] --- diffusion contribution
        _B, _dens, _temp, _mass, _charge, _tene, kz = self.vt.make_value_or_expression(
            self)

        if idx != 0:
            message = "Add diffusion + mass integrator contribution"
            mat = -self._jitted_coeffs["M_perp"]
            if real:
                mat2 = -mat[[0, 1], [0, 1]]
                self.add_integrator(engine, 'diffusion', mat2, a.AddDomainIntegrator,
                                    mfem.DiffusionIntegrator)
                mat2 = (-kz**2)*mat[[2], [2]]
                self.add_integrator(engine, 'mass', mat2, a.AddDomainIntegrator,
                                    mfem.MassIntegrator)
            else:
                mat2 = 1j*kz*mat[[2], [0, 1]]
                self.add_integrator(engine, '12', mat2, a.AddDomainIntegrator,
                                    mfem.MixedDirectionalDerivativeIntegrator)
                mat2 = 1j*kz*mat[[0, 1], [2]]
                self.add_integrator(engine, '21', mat2, a.AddDomainIntegrator,
                                    mfem.MixedScalarWeakDivergenceIntegrator)

            if umode:
                dterm = self._jitted_coeffs["dterms"][idx-1]
            else:
                dterm = self._jitted_coeffs["dterms"][idx-1].conj()
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        else:  # constant term contribution
            message = "Add mass integrator contribution"
            dd0 = self._jitted_coeffs["dd0"]
            self.add_integrator(engine, 'mass', -dd0, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        if real:
            dprint1(message, "(real)",  dep_var, idx)
        else:
            dprint1(message, "(imag)",  dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):

        names = self.current_names_xyz()

        fac = self._jitted_coeffs["fac"]
        M_perp = self._jitted_coeffs["M_perp"]

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

        if c == Exyname or c == Ezname:
            idx, umode, dirc = self._get_dep_var_idx(r, names)
        elif r == Exyname or r == Ezname:
            idx, umode, dirc = self._get_dep_var_idx(c, names)
        else:
            assert False, "Should not come here"

        if c == Exyname:
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if dirc == 'x':
                mat = M_perp[[0], [0, 1]]
            elif dirc == 'y':
                mat = M_perp[[1], [0, 1]]
            elif dirc == 'z':
                mat = M_perp[[2], [0, 1]]

            if umode:
                ccoeff_d = slot["diag"] + slot["diagi"]
                ccoeff2 = mat*ccoeff_d
            else:
                ccoeff = 1j*fac
                ccoeff2 = mat*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)
            return

        if c == Ezname:
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if dirc == 'x':
                mat = M_perp[[0], [2]]
            elif dirc == 'y':
                mat = M_perp[[1], [2]]
            elif dirc == 'z':
                mat = M_perp[[2], [2]]

            if umode:
                ccoeff_d = slot["diag"] + slot["diagi"]
                ccoeff2 = mat*ccoeff_d
            else:
                ccoeff = 1j*fac
                ccoeff2 = mat*ccoeff
            self.add_integrator(engine, 'cterm', ccoeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)
            return

        if r == Exyname and dirc != 'z':
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            coeff2 = mfem.VectorArrayCoefficient(2)
            if umode:
                ccoeff = 1j*fac.conj()
            else:
                ccoeff = (slot["diag"] - slot["diagi"]).conj()

            if real:
                mfem_coeff = ccoeff.get_real_coefficient()
            else:
                mfem_coeff = ccoeff.get_imag_coefficient()

            if dirc == 'x':
                coeff2.Set(0, mfem_coeff)
            elif dirc == 'y':
                coeff2.Set(1, mfem_coeff)
            else:
                assert False, "should not come here"

            coeff2._link = ccoeff

            self.add_integrator(engine, 'cterm', coeff2,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorProductIntegrator)
            return

        if r == Ezname and dirc == 'z':
            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                ccoeff = (1j*fac).conj()
            else:
                ccoeff = (slot["diag"] - slot["diagi"]).conj()

            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)
            return

        dprint1("No mixed-contribution"  "r/c", r, c, is_trans)

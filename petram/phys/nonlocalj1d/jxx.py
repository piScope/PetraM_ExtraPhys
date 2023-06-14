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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jxx')

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
        ('nmax', VtableElement('nmax', type='int',
                               guilabel='nmax',
                               default="3",
                               no_func=True,
                               tip="maximum number of cyclotron harmonics ")),)


class NonlocalJ1D_Jxx(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jxx, self).__init__(**kwargs)

    def count_x_terms(self):
        if not hasattr(self, "_global_ns"):
            return 0
        if not hasattr(self, "_nmax_bk"):
            self._nxterms = 0
            self._nmax_bk = -1

        B, dens, temp, masse, charge, nmax = self.vt.make_value_or_expression(
            self)

        from petram.phys.nonlocalj1d.nonlocalj1d_subs import jxx_terms

        if self._nmax_bk != nmax:
            fits = jxx_terms(nmax=nmax)
            self._approx_computed = True
            total = np.sum([len(fit.c_arr) for fit in fits])
            self._nxterms = total
            self._nmax_bk = nmax

        return int(self._nxterms)

    def get_jx_names(self):
        base = self.get_root_phys().extra_vars_basex
        return [base + self.name() + str(i+1)
                for i in range(self.count_x_terms())]

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        paired_model = self.get_root_phys().paired_model
        mfem_physroot = mm.get_root_phys().parent
        em1d = mfem_physroot[paired_model]

        freq, omega = self.em1d.get_freq_omega()
        ind_vars = self.get_root_phys().ind_vars

        B, dens, temp, mass, charge, nmax = self.vt.make_value_or_expression(
            self)

        from petram.phys.nonlocalj1d.nonlocalj1d_subs import (jxx_terms,
                                                              build_coefficients)

        fits = jxx_terms(nmax=nmax)
        self._jitted_coeffs = build_coefficients(ind_vars, omega, B, dens, temp, mass, charge, fits,
                                                 self._global_ns, self._local_ns,)

    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        Jnl = Jn1 + Jx2 + Jx3 ..
        nabla^2 Jx1 - d1 = c1 * E
        nabla^2 Jx1 - d2 = c2 * E
        nabla^2 Jx3 - d1 = c3 * E
        '''
        Jnlxname = self.get_root_phys().extra_vars_basex
        Jnlterms = self.get_jx_names()

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Ename = var_s[0]

        loc = []
        for n in Jnlterms:
            loc.append((n, Ename, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        from petram.phys.nonlocalj1d.jxx_subs import add_bf_contribution

        root = self.get_root_phys()

        dep_vars = root.dep_vars

        if not root.has_jx:
            return

        jxname = self.get_root_phys().extra_vars_basex

        jnlterms = self.get_jx_names()

        if not dep_vars[kfes] in jnlterms:
            return
        if kfes != 0:
            return
        if dep_vars[kfes][len(jxname)+1:] == '':  # Jnlx
            return

        idx = int(dep_vars[kfes][len(jxname)+1:])  # Jnlx_1, Jnlx_2, ...

        for coeff in self._jitted_coeffs:
            if coeff.n != idx:
                continue
            pass

        if real:
            dprint1("Add diffusion integrator contribution(real)")
        else:
            dprint1("Add diffusion integrator contribution(imag)")

        add_bf_contribution(self, mfem, engine, a, real=real, kfes=kfes)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):
        if real:
            dprint1("Add mixed contribution(real)"  "r/c", r, c, is_trans)
        else:
            dprint1("Add mixed contribution(imag)"  "r/c", r, c, is_trans)

        from petram.phys.nonlocalj1d.jxx_subs import add_mix_contribution2

        add_mix_contribution2(self, mfem, engine, mbf, r, c,  is_trans, _is_conj,
                              real=real)

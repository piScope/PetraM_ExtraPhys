'''

compute non-local current correction.

'''
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
        ('mass', VtableElement('mass', type='array',
                               guilabel='masses(/Da)',
                               default="2, 1",
                               no_func=True,
                               tip="mass. normalized by Da. For electrons, use q_Da")),
        ('charge_q', VtableElement('charge_q', type='array',
                                   guilabel='charges(/q)',
                                   default="1, 1",
                                   no_func=True,
                                   tip="ion charges normalized by q(=1.60217662e-19 [C])")),)


class NonlocalJ1D_Jxx(Domain, Phys):
    has_essential = False
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jxx, self).__init__(**kwargs)

    def count_x_terms(self):
        return 2

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

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
        Jnlnames = [x for x in self.get_root_phys().dep_vars if x.endswith('x')]
        Jnlname = Jnlnames[0]
        Jnlterms = [x for x in self.get_root_phys().dep_vars if x.startswith(Jnlname + "_")]

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

        if kfes != 0:
            return

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

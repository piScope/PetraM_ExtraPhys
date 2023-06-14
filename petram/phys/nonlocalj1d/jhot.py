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
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Jhot')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class NonlocalJ1D_Jhot(NonlocalJ1D_BaseDomain):
    has_essential = False
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True

    def __init__(self, **kwargs):
        super(NonlocalJ1D_Jhot, self).__init__(**kwargs)

    def has_jx(self):
        return int(self.j_direction == 'x')

    def has_jy(self):
        return int(self.j_direction == 'y')

    def has_jz(self):
        return int(self.j_direction == 'z')

    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v["j_direction"] = 'x'
        return v

    def panel1_param(self):
        from wx import CB_READONLY
        panels = super(NonlocalJ1D_Jhot, self).panel1_param()
        panels.append(["direction", "x", 4,
                       {"style": CB_READONLY, "choices": ["x", "y", "z"]}])
        return panels

    def get_panel1_value(self):
        val = super(NonlocalJ1D_Jhot, self).get_panel1_value()
        val.append(self.j_direction)
        return val

    def import_panel1_value(self, v):
        self.j_direction = str(v[-1])

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
        dir = self.j_direction
        Jnlnames = [x for x in self.get_root_phys().dep_vars if x.endswith(dir)]
        Jnlname = Jnlnames[0]
        Jnlterms = [x for x in self.get_root_phys(
        ).dep_vars if x.startswith(Jnlname + "_")]

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_model].dep_vars
        Ename = var_s[0]

        loc = [(Jnlname, Ename, -1, 1)]
        for n in Jnlterms:
            loc.append((n, Jnlname, 1, 1))
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

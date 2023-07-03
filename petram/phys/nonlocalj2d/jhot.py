'''

compute non-local current correction.

'''
from petram.phys.nonlocalj1d.nonlocalj1d_model import NonlocalJ2D_BaseDomain
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_Jhot')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class NonlocalJ2D_Jhot(NonlocalJ2D_BaseDomain):
    has_essential = False
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True

    def __init__(self, **kwargs):
        super(NonlocalJ2D_Jhot, self).__init__(**kwargs)

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
        panels = super(NonlocalJ2D_Jhot, self).panel1_param()
        panels.append(["direction", "x", 4,
                       {"style": CB_READONLY, "choices": ["x", "y", "z"]}])
        return panels

    def get_panel1_value(self):
        val = super(NonlocalJ2D_Jhot, self).get_panel1_value()
        val.append(self.j_direction)
        return val

    def import_panel1_value(self, v):
        self.j_direction = str(v[-1])

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dir = self.j_direction

        if check == 0 and dir == 'x':
            return True
        if check == 1 and dir == 'y':
            return True
        if check == 2 and dir == 'z':
            return True
        return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):

        root = self.get_root_phys()

        paired_model = root.paired_model
        mfem_physroot = root.parent
        var_s = mfem_physroot[paired_model].dep_vars

        dir = self.j_direction
        if dir == 'x':
            base = root.extra_vars_basex
            Ename = var_s[0]
        elif dir == 'y':
            base = root.extra_vars_basey
            Ename = var_s[1]
        elif dir == 'z':
            base = root.extra_vars_basez
            Ename = var_s[2]

        Jnlterms = [x for x in self.get_root_phys().dep_vars
                    if x.startswith(base) and base != x]

        loc = []
        for n in Jnlterms:
            loc.append((base, n, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if real:
            dprint1("Add mass integrator contribution(real)", "kfes=", kfes)
        else:
            return

        sc = mfem.ConstantCoefficient(-1)
        self.add_integrator(engine, 'neg_identity', sc, a.AddDomainIntegrator,
                            mfem.MassIntegrator)

    def add_mix_contribution2(self, engine, mbf, r, c,  is_trans, _is_conj,
                              real=True):
        if real:
            dprint1("Add mixed contribution(real)"  "r/c", r, c)
        else:
            dprint1("Add mixed contribution(imag)"  "r/c", r, c)

        root = self.get_root_phys()

        paired_model = root.paired_model
        mfem_physroot = root.parent
        em1d = mfem_physroot[paired_model]
        var_s = em1d.dep_vars

        dir = self.j_direction
        if dir == 'x':
            base = root.extra_vars_basex
            Ename = var_s[0]
        elif dir == 'y':
            base = root.extra_vars_basey
            Ename = var_s[1]
        elif dir == 'z':
            base = root.extra_vars_basez
            Ename = var_s[2]

        if r == base and c.startswith(base):
            if real:
                sc = mfem.ConstantCoefficient(1)
                self.add_integrator(engine, 'identity', sc,
                                    mbf.AddDomainIntegrator, mfem.MixedScalarMassIntegrator)

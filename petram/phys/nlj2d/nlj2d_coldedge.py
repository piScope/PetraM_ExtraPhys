from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ2D_ColdEdge')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('jx_edge', VtableElement('jx_edge', type='complex',
                                  guilabel='Jx',
                                  default=0.0,
                                  tip="Jx hot correction at edge")),
        ('jy_edge', VtableElement('jy_edge', type='complex',
                                  guilabel='Jy',
                                  default=0.0,
                                  tip="Jy hot correction at edge")),
        ('jz_edge', VtableElement('jz_edge', type='complex',
                                  guilabel='Jz',
                                  default=0.0,
                                  tip="Jz hot correction at edge")),)


def bdry_constraints():
    return [NLJ2D_ColdEdge]


class NLJ2D_ColdEdge(Bdry, Phys):
    has_essential = True
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ2D_ColdEdge, self).__init__(**kwargs)

    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_essential_idx(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dep_var = root.kfes2depvar(kfes)

        if check in (2, 5, 6, 8, 7, 9):
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        jx0, jy0, jz0 = self.vt.make_value_or_expression(self)

        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dep_var = root.kfes2depvar(kfes)

        if check in (2, 5):
            c0 = jz0
            txt = " (z component)"

        elif check in (6, 8):
            c0 = jx0
            txt = " (x component)"

        elif check in (7, 9):
            c0 = jy0
            txt = " (y component)"

        else:
            return

        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index),
                    'kfes', kfes, dep_var, c0, txt)
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index),
                    'kfes', kfes, dep_var, c0, txt)

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = SCoeff([c0], ind_vars, l, g, return_complex=True)
        coeff1 = coeff1.get_realimag_coefficient(real)

        gf.ProjectBdrCoefficient(coeff1,
                                 mfem.intArray(bdr_attr))

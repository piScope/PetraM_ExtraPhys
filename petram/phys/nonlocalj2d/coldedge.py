from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_ColdEdge')

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


class NonlocalJ2D_ColdEdge(Bdry, Phys):
    has_essential = True
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NonlocalJ2D_ColdEdge, self).__init__(**kwargs)

    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_essential_idx(self, kfes):
        # if kfes == 0:
        return self._sel_index
        # else:
        #    return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        jx0, jy0, jz0 = self.vt.make_value_or_expression(self)

        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dep_var = root.kfes2depvar(kfes)

        if check in (0, 3):
            c0 = (jx0, jy0)
            txt = " (xy component)"
            is_xy = True
        elif check in (4,):
            txt = " (p component)"
            return
        elif check in (2, 5):
            c0 = jz0
            txt = " (z component)"
            is_xy = False
        else:
            assert False, 'Unknown check return value'

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

        coeff1 = VCoeff(3, (jx0, jy0, jz0), ind_vars,
                        l, g, return_complex=True)

        if is_xy:
            '''
            coeff1 = Exy(2, Exyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real)
            '''
            coeff1 = coeff1[[0, 1]]
            coeff1 = coeff1.get_realimag_coefficient(real)
            # gf.ProjectBdrCoefficientTangent(coeff1,
            #                                mfem.intArray(bdr_attr))
            gf.ProjectBdrCoefficientNormal(coeff1,
                                           mfem.intArray(bdr_attr))

        else:
            '''
            coeff1 = Ez(Exyz, self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real)
            '''
            coeff1 = coeff1[2]
            coeff1 = coeff1.get_realimag_coefficient(real)
            gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))

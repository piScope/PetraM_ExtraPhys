from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.common.nlj_mixins import NLJ_BaseBdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_ColdEdge')

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
    return [NLJ1D_ColdEdge]


class NLJ1D_ColdEdge(NLJ_BaseBdry):
    has_essential = True
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(NLJ1D_ColdEdge, self).__init__(**kwargs)

    def attribute_set(self, v):
        v = super(NLJ1D_ColdEdge, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_essential_idx(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dep_var = root.kfes2depvar(kfes)

        if check in (18, 19, 20):
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        jx0, jy0, jz0 = self.vt.make_value_or_expression(self)

        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        dep_var = root.kfes2depvar(kfes)
        fes = engine.get_fes(self.get_root_phys(), name=dep_var)

        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index),
                    'kfes:', kfes, dep_var, jx0, jy0, jz0)
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index),
                    'kfes:', kfes, dep_var, jx0, jy0, jz0)

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = VCoeff(3, [jx0, jy0, jz0], ind_vars,
                        l, g, return_complex=True)
        coeff1 = coeff1.get_realimag_coefficient(real)

        ess_vdofs = mfem.intArray()
        fes.GetEssentialVDofs(mfem.intArray(bdr_attr), ess_vdofs, 0)
        vdofs = np.where(np.array(ess_vdofs.ToList()) == -1)[0]
        dofs = mfem.intArray([fes.VDofToDof(i) for i in vdofs])
        fes.BuildDofToArrays()

        gf.ProjectCoefficient(coeff1, dofs)

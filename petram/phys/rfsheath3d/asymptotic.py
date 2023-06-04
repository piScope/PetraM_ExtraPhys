from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('RFsheath3D_Asymptotic')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('label1', VtableElement(None,
                                 guilabel='Asymptotic BC',
                                 default="",
                                 tip="Dn = 0 (curl_t E_t = 0)")),)


class RFsheath3D_Asymptotic(Domain, Phys):
    has_essential = False
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(RFsheath3D_Asymptotic, self).__init__(**kwargs)

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
        [ curl-curl + mass          curl^][  E  ] = [J]
        [ div                  dif       ][ Vsh ] = [0]
        [ curl                           ][ Fmg ] = [0]
        '''
        Vshname = self.get_root_phys().dep_vars[0]
        Fmgname = self.get_root_phys().dep_vars[1]

        paired_var = self.get_root_phys().paired_var
        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[paired_var[0]].dep_vars
        Ename = var_s[paired_var[1]]

        loc = ((Fmgname, Ename, 1, 1),
               (Fmgname, Ename, -1, 1),
               (Vshname, Ename, 1, 1),)
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        from petram.phys.rfsheath3d.asymptotic_subs import add_bf_contribution

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

        from petram.phys.rfsheath3d.asymptotic_subs import add_mix_contribution2

        add_mix_contribution2(self, mfem, engine, mbf, r, c,  is_trans, _is_conj,
                              real=real)

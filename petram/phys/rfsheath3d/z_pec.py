import sys

from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

from petram.phys.rfsheath3d.rfsheath3d_model import RFsheath3D_BaseDomain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('RFsheath3D_ZPec')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('label1', VtableElement(None,
                             guilabel='Sheath Impedance BC',
                             default="",
                             tip="Dn = n x grad Fmd (potential for div-free B)")),
        ('B', VtableElement('bext', type='array',
                          guilabel='magnetic field',
                            default="=[0,0,0]",
        ('temperature', VtableElement('temperature', type='float',
                                      guilabel='temp.(eV)',
                                      default="10.",
                                      tip="temperature ")),
        ('zsh_model', VtableElement('zsh_model', type='string',
                                guilabel='Z(Vsh) model',
                                default="myra17",
                                tip="sheath impedance model")),
        ('smoothing', VtableElement('smoothing', type='float',
                                guilabel='Smoothing (a_s)',
                                default="0.0",
                                tip="smoothing factor Eq.24 in SS NF2023.")),)

try:
    from petram.phys.rfsheath3d.rfsheath3d_subs import (zpec_get_mixedbf_loc,
                                                        zpec_add_bf_contribution,
                                                        zpec_add_mix_contribution2)

    get_mixedbf_loc=zpec_get_mixedbf_loc
    add_bf_contribution=zpec_add_bf_contribution
    add_mix_contribution2=zpec_add_mix_contribution2
except ImportError:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "rfsheath"]:
        sys.modules[__name__].dependency_invalid=True


class RFsheath3D_ZPec(RFsheath3D_BaseDomain):
    has_essential=False
    nlterms=[]
    can_timedpendent=True
    has_3rd_panel=True
    vt=Vtable(data)

    def __init__(self, **kwargs):
        super(RFsheath3D_ZPec, self).__init__(**kwargs)

    @ classmethod
    def fancy_menu_name(cls):
        return "Impedance"

    @ classmethod
    def fancy_tree_name(cls):
        return "Impedance"

    @ property
    def is_nonlinear(self):
        return True

    def attribute_set(self, v):
        RFsheath3D_BaseDomain.attribute_set(self, v)
        v['sel_readonly']=False
        v['sel_index']=[]
        return v

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        return get_mixedbf_loc(self)

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

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


        label, B, temp, zsh, alpha=self.vt.make_value_or_expression(self)
        g_ns=self._global_ns
        l_ns=self._local_ns


        add_mix_contribution2(self, mfem, engine, mbf, r, c,  is_trans, _is_conj,
                              real=real, params=(B, temp, zsh, alpha, l_ns, g_ns))

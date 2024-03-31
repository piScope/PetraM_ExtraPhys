import sys

from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

from petram.phys.rfsheath3d.rfsheath3d_model import RFsheath3D_BaseDomain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('RFsheath3D_Asymptotic')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('label1', VtableElement(None,
                                 guilabel='Asymptotic BC',
                                 default="",
                                 tip="Dn = 0 (curl_t E_t = 0)")),
        ('B', VtableElement('bext', type='array',
                            guilabel='magnetic field',
                            default="=[0,0,0]",
                            tip="static magnetic field")),
        ('temperature', VtableElement('temperature', type='float',
                                      guilabel='temp.(eV)',
                                      default="10.",
                                      tip="temperature ")),
        ('zsh_model', VtableElement('zsh_model', type='string',
                                    guilabel='Vdc model',
                                    default="myra17",
                                    tip="sheath impedance model used to compute Vdc")),)

try:
    from petram.phys.rfsheath3d.rfsheath3d_subs import (asymptotic_get_mixedbf_loc,
                                                        asymptotic_add_bf_contribution,
                                                        asymptotic_add_mix_contribution2)
    get_mixedbf_loc = asymptotic_get_mixedbf_loc
    add_bf_contribution = asymptotic_add_bf_contribution
    add_mix_contribution2 = asymptotic_add_mix_contribution2
except ImportError:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "rfsheath"]:
        sys.modules[__name__].dependency_invalid = True


class RFsheath3D_Asymptotic(RFsheath3D_BaseDomain):
    has_essential = False
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True
    vt = Vtable(data)

    def __init__(self, **kwargs):
        super(RFsheath3D_Asymptotic, self).__init__(**kwargs)

    def attribute_set(self, v):
        RFsheath3D_BaseDomain.attribute_set(self, v)
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

        add_mix_contribution2(self, mfem, engine, mbf, r, c,  is_trans, _is_conj,
                              real=real)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import NativeCoefficientGenBase

        if len(self._sel_index) == 0:
            return

        paired_model = self.get_root_phys().paired_model
        mfem_physroot = self.get_root_phys().parent
        em3d = mfem_physroot[paired_model]
        freq, omega = em3d.get_freq_omega()

        _label, mag, temp, zsh = self.vt.make_value_or_expression(self)
        print("checking ", mag, temp, zsh)

from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.nlj1d.nlj1d_model import NLJ1D_BaseBdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_Continuity')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def bdry_constraints():
    return [NLJ1D_Continuity]


class NLJ1D_Continuity(NLJ1D_BaseBdry):
    is_essential = False

    def __init__(self, **kwargs):
        super(NLJ1D_Continuity, self).__init__(**kwargs)
        self.sel_readonly = False
        self.sel_index = []

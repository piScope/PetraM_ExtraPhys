from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.phys_cont import PhysContinuity

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ2D_Continuity')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def bdry_constraints():
    return [NLJ2D_Continuity]


class NLJ2D_Continuity(PhysContinuity):
    pass

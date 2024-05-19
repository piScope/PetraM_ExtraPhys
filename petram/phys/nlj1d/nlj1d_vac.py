from petram.mfem_config import use_parallel
from petram.phys.common.nlj_mixins import NLJ_Vac
from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable

import numpy as np


import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_Vac')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def domain_constraints():
    return [NLJ1D_Vac]


class NLJ1D_Vac(NLJ_Vac):
    
    has_essential = False
    nlterms = []
    has_3rd_panel = True
    
    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        
        if not real:
            return

        if not real:
            return
        
        dterm = mfem.VectorConstantCoefficient([1, 1, 1.])
        self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                            mfem.VectorMassIntegrator,)
        
        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)
        
        message = "Add mass integrator contribution"
        dprint1(message, "(real)", dep_var, str(self._sel_index))        

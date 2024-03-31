'''

helper to handle H1 vectors


    (u, M v) : 
    (div R1u, div R2v)
    (curl R1u, curl R2v)

'''
from petram.phys.numba_coefficient import NumbaCoefficient
from petram.mfem_config import use_parallel
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('VectorFE_helper_mixin')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class VectorFEHelper_mxin():
    def fill_mass_matrix(self, engine, form, i, j, m):
        '''
        (A, m B) : A vector, B vector

        M : matrix, vector (treated as diagnal matrix), or scalar
        '''

        if isinstance(m, NumbaCoefficient):
            if len(m.shape) == 2:
                coeff1 = m[i, j]
            elif len(m.shape) == 1:
                if i != j:
                    return
                coeff1 = m[i]
            elif len(m.shape) == 0:
                coeff1 = m
            else:
                assert False, "Tensor Coefficient not support"
        elif isinstance(m, mfem.Coefficient):
            coeff1 = m
        else:
            print("input", m)
            assert False, "unsupported coefficient"

        self.add_integrator(engine, 'cterm', coeff1,
                            form.AddDomainIntegrator,
                            mfem.MixedScalarMassIntegrator)

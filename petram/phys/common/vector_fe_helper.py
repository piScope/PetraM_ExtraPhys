'''

helper to handle H1 vectors


    (u, M v) : 
    (div R1u, div R2v)
    (curl R1u, curl R2v)

'''
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('VectorFE_helper_mixin')

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

    
    
class VectorFEHelper_mxin():
    def fill_mass_matrix(self, engine, form, i, j, m):
        '''
        (A, m B) : A vector, B vector
        '''
        coeff1 = coeff[i,j]

        self.add_integrator(engine, 'cterm', coeff1,
                           form.AddDomainIntegrator,
                           mfem.MixedScalarMassIntegrator)

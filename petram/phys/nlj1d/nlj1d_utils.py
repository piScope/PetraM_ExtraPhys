import numpy as np
from petram.helper.variables import variable
delta = 0.001

def simple_jacB_1D(f):
    def b_dot(x):
        '''
        return B[i,j] = dBi/dxj = \partial_j b_i

        '''
        ret = np.zeros((3, 3), dtype=np.float64)
        B = f(x+delta/2.) - f(x-delta/2.)
        B /= delta
        ret[0, 0] = B[0]
        ret[1, 0] = B[1]
        ret[2, 0] = B[2]
        
        return ret.flatten()
    ret = variable.jit.array(shape=(9,))(b_dot)
    return ret

def simple_hessB_1D(f):
    def b_dot2(x):    
        '''
        return B[i,j, k] = dBi/dxj/dxk = \partial_j \partial_k b_i

        '''
        ret = np.zeros((3, 3, 3), dtype=np.float64)
        B = f(x+delta) +f(x-delta) - 2*f(x)
        B /= delta
        B /= delta

        ret[0, 0, 0] = B[0]
        ret[1, 0, 0] = B[1]
        ret[2, 0, 0] = B[2]

        return ret.flatten()
    ret = variable.jit.array(shape=(27,))(b_dot2)
    return ret

    
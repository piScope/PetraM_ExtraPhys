import numpy as np
from petram.helper.variables import variable
delta = 0.001


def simple_jacB_2D(f):
    def b_dot(x, y):
        '''
        return B[i,j] = dBi/dxj = \partial_j b_i

        '''
        ret = np.zeros((3, 3), dtype=np.float64)
        dBdx = f(x+delta/2., y) - f(x-delta/2., y)
        dBdx /= delta
        dBdy = f(x, y+delta/2.) - f(x, y-delta/2.)
        dBdy /= delta
        ret[0, 0] = dBdx[0]
        ret[1, 0] = dBdx[1]
        ret[2, 0] = dBdx[2]
        ret[0, 1] = dBdy[0]
        ret[1, 1] = dBdy[1]
        ret[2, 1] = dBdy[2]

        return ret.flatten()
    ret = variable.jit.array(shape=(9,))(b_dot)
    return ret


def simple_hessB_2D(f):
    def b_dot2(x, y):
        '''
        return B[i,j, k] = dBi/dxj/dxk = \partial_j \partial_k b_i

        '''
        ret = np.zeros((3, 3, 3), dtype=np.float64)
        dBdxx = f(x+delta, y) + f(x-delta, y) - 2*f(x, y)
        dBdxx /= delta
        dBdxx /= delta
        dBdyy = f(x, y+delta) + f(x, y-delta) - 2*f(x, y)
        dBdyy /= delta
        dBdyy /= delta
        dBdxy = f(x+delta, y+delta) - f(x+delta, y-delta) - \
            f(x-delta, y+delta) + f(x-delta, y-delta)
        dBdx /= 2*delta
        dBdxy /= 2*delta

        ret[0, 0, 0] = dBdxx[0]
        ret[1, 0, 0] = dBdxx[1]
        ret[2, 0, 0] = dBdxx[2]
        ret[0, 0, 1] = dBdxy[0]
        ret[1, 0, 1] = dBdxy[1]
        ret[2, 0, 1] = dBdxy[2]
        ret[0, 1, 0] = dBdxy[0]
        ret[1, 1, 0] = dBdxy[1]
        ret[2, 1, 0] = dBdxy[2]
        ret[0, 1, 1] = dBdyy[0]
        ret[1, 1, 1] = dBdyy[1]
        ret[2, 1, 1] = dBdyy[2]

        return ret.flatten()
    ret = variable.jit.array(shape=(27,))(b_dot2)
    return ret
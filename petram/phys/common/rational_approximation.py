'''
   rational approximation utility

     find_decomposition(func, x, xp=None, viewer=None, mmin=2,  mmax=5, **kwargs)

       find an approximation for function <func> over x range given by <x>

       plot option:
           xp : x range used in plotting
           viewer : a viewer used for plotting. Typically give figure()

       number of terms:  mmin - mmax

'''
import numpy as np
from baryrat import aaa

try:
    from numba import njit, void, int64, float64, complex128, types
    use_numba = True
except ImportError:
    use_numba = False


def P(r):
    '''
    compute numerator of approximation f(x) = P(x)/Q(x)
    '''
    z = r.nodes
    w = r.weights
    f = r.values

    nterms = len(f)

    poly = np.poly1d([])

    for i in range(nterms):
        tmp = np.poly1d([f[i]*w[i]])
        for j in range(nterms):
            if i == j:
                continue
            tmp = tmp * np.poly1d([1, -z[j]])
        poly = poly + tmp

    return poly


def Q(r):
    '''
    compute denominaator of approximation f(x) = P(x)/Q(x)
    '''
    z = r.nodes
    w = r.weights
    f = r.values

    nterms = len(f)

    poly = np.poly1d([])

    for i in range(nterms):
        tmp = np.poly1d([w[i]])
        for j in range(nterms):
            if i == j:
                continue
            tmp = tmp * np.poly1d([1, -z[j]])

        poly = poly + tmp
    return poly


class poly_fraction():
    def __init__(self, c0, c_arr, d_arr):
        self.c0 = c0
        self.c_arr = c_arr
        self.d_arr = d_arr

    def __call__(self, x):
        value = np.zeros(x.shape, dtype=np.complex128) + self.c0
        for c, d in zip(self.c_arr, self.d_arr):
            value = value + c / (x - d)

        return value

    def __repr__(self):
        txt = ["fractional sum",
               "c0: " + str(self.c0),
               "c and d: "]

        for c, d in zip(self.c_arr, self.d_arr):
            txt.append(str((c, d)))
        return "\n".join(txt)


def calc_decomposition(func, x, mmax, xp=None, viewer=None, **kwargs):
    if xp is None:
        xp = x

    f = np.array([func(xx, **kwargs) for xx in x])

    if viewer is not None:
        fp = np.array([func(xx, **kwargs) for xx in xp])
        viewer.plot(xp, fp, 'r')
        if np.iscomplexobj(f):
            viewer.plot(xp, fp.imag, 'b')
        viewer.xlabel("x")

    r = aaa(x, f, tol=0, mmax=mmax)
    # if viewer is not None:
    #    viewer.plot(x, r(x), 'ro')
    #    if np.iscomplexobj(f):
    #        viewer.plot(x, r(x).imag, 'bo')
    poly_p = P(r)
    poly_q = Q(r)
    poly_q_prime = poly_q.deriv()

    roots = np.roots(poly_q)

    c_arr = []
    d_arr = []
    for root in roots:
        c_arr.append(poly_p(root)/poly_q_prime(root))
        d_arr.append(root)

    c_arr = np.array(c_arr)
    d_arr = np.array(d_arr)

    if not np.all([d.real < 0 for d in roots]):
        return None

    tmp = 0j
    for c, d in zip(c_arr, d_arr):
        tmp = tmp + c/(x[0]-d)

    c0 = func(x[0], **kwargs)-tmp

    if use_numba:
        @njit(complex128(float64))
        def f_sum(x):
            value = c0+0j
            for i in range(len(c_arr)):
                value = value + c_arr[i] / (x - d_arr[i])
            # for c, d in zip(c_arr, d_arr):
            #    value = value + c / (x - d)
            return value
        f_sum.c_arr = c_arr
        f_sum.d_arr = d_arr
        f_sum.c0 = c0
    else:
        f_sum = poly_fraction(c0, c_arr, d_arr)

    fit = np.array([f_sum(xx) for xx in xp])
    if viewer is not None:
        viewer.plot(xp, fit, 'g--')
        if np.iscomplexobj(f):
            viewer.plot(xp, fit.imag, 'g--')

    return f_sum


def find_decomposition(func, x, xp=None, viewer=None, mmin=2, mmax=8, **kwargs):
    mm = mmin
    while mm <= mmax:
        fit = calc_decomposition(func, x, mm, xp=xp, viewer=viewer, **kwargs)
        if fit is not None:
            break

        mm = mm + 1

    success = np.all([d.real < 0 for d in fit.d_arr])

    return fit, success

import numpy as np
from baryrat import aaa

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
    def __init__(self, c0, c_and_d):
        self.c0 = c0
        self.c_and_d = c_and_d

    def __call__(self, x):
        value = np.zeros(x.shape, dtype=np.complex128) + self.c0
        for c, d in self.c_and_d:
            value = value + c / (x - d)

        return value

    def __repr__(self):
        txt = ["fractional sum",
               "c0: " + str(self.c0),
               "c and d: "]

        for x in self.c_and_d:
            txt.append(str(x))
        return "\n".join(txt)


def calc_decomposition(func, x, mmax, xp, viewer=None, **kwargs):
    f = np.array([func(xx, **kwargs) for xx in x])

    if viewer is not None:
        fp = np.array([func(xx, **kwargs) for xx in xp])        
        viewer.plot(xp, fp, 'r')
        if np.iscomplexobj(f):
            viewer.plot(xp, fp.imag, 'b')
        viewer.xlabel("x")


    r = aaa(x, f, tol=0, mmax=mmax)
    #if viewer is not None:
    #    viewer.plot(x, r(x), 'ro')
    #    if np.iscomplexobj(f):
    #        viewer.plot(x, r(x).imag, 'bo')
    poly_p = P(r)
    poly_q = Q(r)
    poly_q_prime = poly_q.deriv()

    roots = np.roots(poly_q)

    c_and_d = []
    for root in roots:
        c_and_d.append((poly_p(root)/poly_q_prime(root), root))
        
    if not np.all([d.real<0 for d in roots]):
        return None
    
    tmp = 0j
    for c, d in c_and_d:
        tmp = tmp + c/(x[0]-d)

    print(x[0], func(x[0], **kwargs), tmp)
    c0 = func(x[0], **kwargs)-tmp

    f_sum = poly_fraction(c0, c_and_d)

    fit = np.array([f_sum(xx) for xx in xp])

    if viewer is not None:
        viewer.plot(xp, fit, 'g--')
        if np.iscomplexobj(f):
            viewer.plot(xp, fit.imag, 'g--')

    print(f_sum)
    return f_sum

def find_decomposition(func, x, xp, viewer=None, mmax_min=2,  mmax_max=5, **kwargs):
    mm = mmax_min
    while mm <= mmax_max:
        fit = calc_decomposition(func, x, mm, xp, viewer=viewer, **kwargs)
        if fit is not None:
            break

        mm = mm + 1
    
    success = np.all([d.real<0 for c, d in fit.c_and_d])

    return fit, success
    
        

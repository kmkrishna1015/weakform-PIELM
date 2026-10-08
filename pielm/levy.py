"""Exact frequencies of a unit square Kirchhoff plate, x=0,1 simply supported, y=0,1 FREE (Levy solution), nu=0.3.
w = sin(m pi x) Y(y);  Y'''' - 2 a2 Y'' + (a2^2 - Om^2) Y = 0,  a2=(m pi)^2 ;  free edges:  Y'' - nu a2 Y = 0,  Y''' - (2-nu) a2 Y' = 0."""
import mpmath as mp, numpy as np
mp.mp.dps = 40
def det(Om, m, nu=0.3):
    a2 = (m * mp.pi) ** 2; g2 = Om + a2; Om = mp.mpf(Om); g = mp.sqrt(g2); d2 = Om - a2
    if d2 > 0: d = mp.sqrt(d2); fs = [lambda y: (mp.cosh(g*y), g*mp.sinh(g*y), g**2*mp.cosh(g*y), g**3*mp.sinh(g*y)), lambda y: (mp.sinh(g*y), g*mp.cosh(g*y), g**2*mp.sinh(g*y), g**3*mp.cosh(g*y)),
                          lambda y: (mp.cos(d*y), -d*mp.sin(d*y), -d**2*mp.cos(d*y), d**3*mp.sin(d*y)), lambda y: (mp.sin(d*y), d*mp.cos(d*y), -d**2*mp.sin(d*y), -d**3*mp.cos(d*y))]
    else: d = mp.sqrt(-d2); fs = [lambda y: (mp.cosh(g*y), g*mp.sinh(g*y), g**2*mp.cosh(g*y), g**3*mp.sinh(g*y)), lambda y: (mp.sinh(g*y), g*mp.cosh(g*y), g**2*mp.sinh(g*y), g**3*mp.cosh(g*y)),
                            lambda y: (mp.cosh(d*y), d*mp.sinh(d*y), d**2*mp.cosh(d*y), d**3*mp.sinh(d*y)), lambda y: (mp.sinh(d*y), d*mp.cosh(d*y), d**2*mp.sinh(d*y), d**3*mp.cosh(d*y))]
    M = mp.zeros(4, 4)
    for r, y in enumerate((0, 1)):
        for c, f in enumerate(fs):
            v = f(y); M[2*r, c] = v[2] - nu * a2 * v[0]; M[2*r+1, c] = v[3] - (2 - nu) * a2 * v[1]
    return mp.det(M)
def levy_sfsf(n=6, nu=0.3, mmax=7):
    out = []
    for m in range(1, mmax + 1):
        prev = None; Om = mp.mpf("0.3")
        while Om < 90:
            v = det(Om, m, nu)
            if prev is not None and v * prev < 0 and abs(Om - mp.mpf(m * m) * mp.pi ** 2 ) > 0: 
                r = mp.findroot(lambda z: det(z, m, nu), (Om - mp.mpf("0.02"), Om), solver="illinois", tol=1e-25, verify=False, maxsteps=200); out.append((float(r) , m))
            prev = v; Om += mp.mpf("0.02")
    out.sort(); return out[:n]
if __name__ == "__main__":
    r = levy_sfsf(); print(r)

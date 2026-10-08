"""1-D frozen bases with a common interface  .d(x, k) -> [phi_j^(k)(x)],  .nphi,  .L."""
import math
import numpy as np


class Bern1D:
    """Bernstein basis of degree n on [0, L] with analytic derivatives (difference recursion)."""
    name = "bernstein"
    def __init__(self, L, n):
        self.L, self.n, self.nphi = float(L), int(n), int(n) + 1
        self._c = {m: np.array([math.comb(m, k) for k in range(m + 1)], float) for m in range(n + 1)}
    def _b(self, t, m):
        i = np.arange(m + 1)
        return self._c[m] * t[:, None] ** i * (1.0 - t[:, None]) ** (m - i)
    def _d(self, t, m, k):
        if k == 0:
            return self._b(t, m)
        lo = self._d(t, m - 1, k - 1); z = np.zeros((lo.shape[0], 1))
        return m * (np.hstack([z, lo]) - np.hstack([lo, z]))
    def d(self, x, k=0):
        t = np.atleast_1d(np.asarray(x, float)) / self.L
        return self._d(t, self.n, k) / self.L ** k


class Leg1D:
    name = "legendre"
    def __init__(self, L, n): self.L, self.n, self.nphi, self._C = float(L), int(n), int(n) + 1, {}
    def d(self, x, k=0):
        from numpy.polynomial import legendre as Lg
        s = 2.0 * np.atleast_1d(np.asarray(x, float)) / self.L - 1.0
        if k == 0: return Lg.legvander(s, self.n)
        if k not in self._C: self._C[k] = Lg.legder(np.eye(self.n + 1), k, axis=0)
        return Lg.legvander(s, self.n - k) @ self._C[k] * (2.0 / self.L) ** k


class Cheb1D:
    name = "chebyshev"
    def __init__(self, L, n): self.L, self.n, self.nphi, self._C = float(L), int(n), int(n) + 1, {}
    def d(self, x, k=0):
        from numpy.polynomial import chebyshev as C
        s = 2 * np.atleast_1d(np.asarray(x, float)) / self.L - 1
        if k == 0: return C.chebvander(s, self.n)
        if k not in self._C: self._C[k] = C.chebder(np.eye(self.n + 1), k, axis=0)
        return C.chebvander(s, self.n - k) @ self._C[k] * (2 / self.L) ** k


class Mono1D:
    name = "monomial"
    def __init__(self, L, n): self.L, self.n, self.nphi = float(L), int(n), int(n) + 1
    def d(self, x, k=0):
        t = np.atleast_1d(np.asarray(x, float)) / self.L; j = np.arange(self.n + 1); ff = np.ones(self.n + 1)
        for q in range(k): ff = ff * (j - q)
        e = np.clip(j - k, 0, None)
        return (ff * t[:, None] ** e) * (j >= k) / self.L ** k


class RandTanh1D:
    """phi_j(x) = tanh(a_j (x/L - c_j)), c_j ~ U(0,1), a_j ~ U(-R,R), drawn once (the ELM hidden layer)."""
    name = "tanh-ELM"
    def __init__(self, L, nphi, R=20.0, seed=0):
        r = np.random.default_rng(seed); self.L, self.nphi = float(L), int(nphi)
        self.a = r.uniform(-R, R, nphi); self.a[np.abs(self.a) < 1e-3] = 1.0; self.c = r.uniform(0, 1, nphi)
    def d(self, x, k=0):
        t = np.tanh(self.a * (np.atleast_1d(np.asarray(x, float))[:, None] / self.L - self.c)); u = 1 - t * t; a = self.a / self.L
        if k == 0: return t
        if k == 1: return a * u
        if k == 2: return a ** 2 * (-2 * t * u)
        if k == 3: return a ** 3 * u * (6 * t * t - 2)
        if k == 4: return a ** 4 * u * (16 * t - 24 * t ** 3)
        raise ValueError(k)


class RBF1D:
    """Gaussian bumps, uniform centres on [0, L], width = ovl * spacing."""
    name = "gaussian-RBF"
    def __init__(self, L, nphi, ovl=2.0):
        self.L, self.nphi = float(L), int(nphi); self.c = np.linspace(0, L, nphi); self.w = ovl * L / max(nphi - 1, 1)
    def d(self, x, k=0):
        u = (np.atleast_1d(np.asarray(x, float))[:, None] - self.c) / self.w; g = np.exp(-0.5 * u * u)
        He = [np.ones_like(u), u, u * u - 1, u ** 3 - 3 * u, u ** 4 - 6 * u * u + 3][k]
        return (-1 / self.w) ** k * He * g


class ADBern1D:
    """Bernstein basis with derivatives by nested torch forward-mode AD (needs `torch`; timing study only)."""
    name = "bernstein-AD"
    def __init__(self, L, n, max_order=4):
        import torch
        from torch.func import jacfwd, vmap
        torch.set_default_dtype(torch.float64)
        self.torch, self.vmap = torch, vmap
        self.L, self.n, self.nphi = float(L), int(n), int(n) + 1
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)], dtype=torch.float64)
        idx = torch.arange(n + 1, dtype=torch.float64); Lf = self.L
        def phi(x):
            t = x / Lf
            return comb * t ** idx * (1.0 - t) ** (n - idx)
        self.fns = [phi]
        for _ in range(max_order): self.fns.append(jacfwd(self.fns[-1]))
        w = torch.linspace(0.0, Lf, 4, dtype=torch.float64)
        for f in self.fns: vmap(f)(w)                                  # warm-up (graph tracing)
    def d(self, x, k=0):
        t = self.torch.tensor(np.atleast_1d(np.asarray(x, float)), dtype=self.torch.float64)
        return self.vmap(self.fns[k])(t).numpy()


def make_basis(kind, L, n, max_order=4):
    """Bernstein / Legendre of degree n (nphi = n+1), or the autodiff Bernstein basis."""
    if kind == "ad": return ADBern1D(L, n, max_order)
    return {"bernstein": Bern1D, "legendre": Leg1D}[kind](L, n)


def make_extra(kind, L, nphi, seed=0, **kw):
    """Basis-study factory: every basis addressed by its number of functions nphi."""
    if kind == "bernstein": return Bern1D(L, nphi - 1)
    if kind == "legendre": return Leg1D(L, nphi - 1)
    if kind == "chebyshev": return Cheb1D(L, nphi - 1)
    if kind == "monomial": return Mono1D(L, nphi - 1)
    if kind == "tanh-ELM": return RandTanh1D(L, nphi, seed=seed, **kw)
    if kind == "gaussian-RBF": return RBF1D(L, nphi, **kw)
    raise KeyError(kind)

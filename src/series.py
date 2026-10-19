"""Herramientas de series de tiempo hechas a mano con numpy.

La misma autocorrelacion y el mismo Ljung-Box se usan en la Parte 0 (sobre
miles de replicas a la vez) y en las Partes 3 y 4 (sobre la serie real).
"""

import numpy as np
from scipy import stats


def acf(y, kmax):
    """r_1, ..., r_kmax. Acepta un vector o una matriz con una serie por fila.

    r_k = sum_{t=k+1}^n (y_t - ybar)(y_{t-k} - ybar) / sum_{t=1}^n (y_t - ybar)^2
    """
    y = np.asarray(y, dtype=float)
    vector = y.ndim == 1
    y = np.atleast_2d(y)
    d = y - y.mean(axis=1, keepdims=True)
    den = (d * d).sum(axis=1)
    r = np.empty((y.shape[0], kmax))
    for k in range(1, kmax + 1):
        r[:, k - 1] = (d[:, k:] * d[:, :-k]).sum(axis=1) / den
    return r[0] if vector else r


def ljung_box(y, m, gl_menos=0):
    """Q(m) = n(n+2) sum_{k=1}^m r_k^2/(n-k), con su valor-p chi2(m - gl_menos).

    gl_menos es el numero de parametros estimados cuando se aplica a residuos.
    Acepta una matriz con una serie por fila.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    r = acf(y, m)
    k = np.arange(1, m + 1)
    q = n * (n + 2) * (r ** 2 / (n - k)).sum(axis=-1)
    p = stats.chi2.sf(q, m - gl_menos)
    return q, p


def ols_ar1(y):
    """OLS de y_t contra y_{t-1} con intercepto. Devuelve (c, phi).

    Acepta una matriz con una serie por fila (una regresion por fila).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    x, z = y[:, :-1], y[:, 1:]
    xm, zm = x.mean(axis=1, keepdims=True), z.mean(axis=1, keepdims=True)
    phi = ((x - xm) * (z - zm)).sum(axis=1) / ((x - xm) ** 2).sum(axis=1)
    c = zm[:, 0] - phi * xm[:, 0]
    if c.size == 1:
        return float(c[0]), float(phi[0])
    return c, phi

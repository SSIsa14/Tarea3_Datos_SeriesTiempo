"""Parte 4: del AR(1) al SARIMA.

Sobre los dias abiertos del entrenamiento (una fila por dia): OLS contra
indicadoras de dia de la semana, AR(1) sobre esos residuos y los dos SARIMA
con s = 6 sobre el logaritmo de la venta, con su diagnostico de residuos.

Uso:  python3 src/parte4_sarima.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, CIRUELA, GRIS,
                   escribir_tex, macro, libras, cientifica, guardar, subtitulo,
                   bandas_acf, piruleta)
from datos import serie_trabajo, abiertos, FIN_ENTRENA
from series import acf, ljung_box, ols_ar1

KMAX = 21
MAXITER = 500

MODELOS = {
    "S1": dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c"),
    "S2": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend="n"),
}


def ajustar(y_log, nombre, maxiter=MAXITER):
    """Ajusta S1 o S2 sobre el arreglo de valores (sin fechas)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y_log, **MODELOS[nombre]).fit(disp=False, maxiter=maxiter)


def indicadoras(fechas):
    """Una columna por dia abierto de la semana (sin sabado), sin intercepto."""
    dias = [0, 1, 2, 3, 4, 6]
    return np.column_stack([(fechas.dayofweek == d).astype(float) for d in dias])


def figura_acfs(paneles, n):
    fig, ejes = plt.subplots(2, 2, figsize=(10, 5.4), sharey=True)
    k = np.arange(1, KMAX + 1)
    for ax, (titulo, sub, r) in zip(ejes.flat, paneles):
        banda = bandas_acf(ax, n, KMAX)
        piruleta(ax, k, r, banda)
        for j in (6, 12, 18):
            ax.axvline(j, color=MOSTAZA, lw=0.8, ls=":", zorder=0)
        ax.set_xticks(k)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(titulo)
        subtitulo(ax, sub)
    for ax in ejes[1]:
        ax.set_xlabel("rezago k (días abiertos)")
    for ax in ejes[:, 0]:
        ax.set_ylabel("$r_k$")
    fig.tight_layout(h_pad=1.6)
    return guardar(fig, "p4_acf_residuos.png")


def figura_horizonte(fechas, y, fc1, fc2, h):
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    hist = 60
    x_hist = np.arange(-hist + 1, 1)
    x_fc = np.arange(1, h + 1)
    ax.plot(x_hist, y[-hist:] / 1000, color=GRIS, lw=1, marker="o", ms=2.2,
            label="venta observada")
    ax.plot(x_fc, fc1 / 1000, color=PETROLEO, lw=1.6, label="S1 (estacionario)")
    ax.plot(x_fc, fc2 / 1000, color=TERRACOTA, lw=1.6, label="S2 (con diferencias)")
    nivel = np.exp(MEDIA_S1) / 1000
    ax.axhline(nivel, color=PETROLEO, lw=0.8, ls="--")
    ax.text(h, nivel, "media de S1 ", color=PETROLEO, fontsize=7.5, ha="right", va="bottom")
    ax.axvline(0.5, color=TINTA, lw=0.8)
    ax.set_xlabel("posición respecto al 30/09/2011 (días abiertos)")
    ax.set_ylabel("miles de £")
    ax.set_title("¿Hacia dónde van los pronósticos cuando el horizonte crece?")
    subtitulo(ax, f"ajuste con todo el entrenamiento, pronóstico a {h} días abiertos")
    ax.legend(loc="upper left", ncol=3)
    return guardar(fig, "p4_horizonte.png")


MEDIA_S1 = None


def main():
    global MEDIA_S1
    s = serie_trabajo()
    ent = abiertos(s[s.index <= FIN_ENTRENA])
    y = ent.values
    n = len(y)

    # 4.1 OLS con indicadoras de dia de la semana
    X = indicadoras(ent.index)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    r_e = acf(e, KMAX)
    q6_e, p6_e = ljung_box(e, 6)
    q12_e, p12_e = ljung_box(e, 12)
    print(f"4.1 r1..r7 de e_t: {np.round(r_e[:7], 3)}  Q6={q6_e:.1f} Q12={q12_e:.1f}")

    # 4.2 AR(1) sobre los residuos
    c, phi = ols_ar1(e)
    u = e[1:] - c - phi * e[:-1]
    r_u = acf(u, KMAX)
    banda = 1.959964 / np.sqrt(len(u))
    fuera_u = (np.flatnonzero(np.abs(r_u) > banda) + 1).tolist()
    q6_u, p6_u = ljung_box(u, 6, gl_menos=1)
    q12_u, p12_u = ljung_box(u, 12, gl_menos=1)
    print(f"4.2 phi={phi:.3f}  fuera={fuera_u}  Q6={q6_u:.1f} p={p6_u:.3g}  "
          f"Q12={q12_u:.1f} p={p12_u:.3g}")

    # 4.3 SARIMA sobre el logaritmo
    y_log = np.log(y)
    filas_conv = []
    ajustes = {}
    for nombre in MODELOS:
        corto = ajustar(y_log, nombre, maxiter=50)
        largo = ajustar(y_log, nombre, maxiter=MAXITER)
        ajustes[nombre] = largo
        for etiqueta, res in (("50 (defecto)", corto), (f"{MAXITER}", largo)):
            params = ", ".join(f"{p:.3f}" for p in res.params)
            conv = "sí" if res.mle_retvals["converged"] else "no"
            filas_conv.append(f"{nombre} & {etiqueta} & {conv} & "
                              f"${res.mle_retvals['iterations']}$ & ${params}$ \\\\")
            print(f"4.3 {nombre} maxiter={etiqueta:<12} convergio={conv} "
                  f"params={dict(zip(res.model.param_names, np.round(res.params, 4)))}")
    escribir_tex("tabla_p4_convergencia.tex", filas_conv)
    nombres_s1 = ", ".join(ajustes["S1"].model.param_names)
    nombres_s2 = ", ".join(ajustes["S2"].model.param_names)

    # 4.4 diagnostico de residuos
    res1 = ajustes["S1"].resid
    res2 = ajustes["S2"].resid[7:]
    diag = {}
    for nombre, rr in (("S1", res1), ("S2", res2)):
        q6, p6 = ljung_box(rr, 6, gl_menos=2)
        q12, p12 = ljung_box(rr, 12, gl_menos=2)
        diag[nombre] = (q6, p6, q12, p12)
        print(f"4.4 {nombre}: Q6={q6:.2f} p={p6:.3f}  Q12={q12:.2f} p={p12:.3f}")

    filas_lb = [
        f"Indicadoras ($e_t$) & ${q6_e:.1f}$ & ${cientifica(p6_e)}$ & ${q12_e:.1f}$ & ${cientifica(p12_e)}$ & 6 / 12 \\\\",
        f"AR(1) sobre $e_t$ ($u_t$) & ${q6_u:.1f}$ & ${cientifica(p6_u)}$ & ${q12_u:.1f}$ & ${cientifica(p12_u)}$ & 5 / 11 \\\\",
    ]
    for nombre in ("S1", "S2"):
        q6, p6, q12, p12 = diag[nombre]
        filas_lb.append(f"{nombre} & ${q6:.2f}$ & ${p6:.3f}$ & ${q12:.2f}$ & ${p12:.3f}$ & 4 / 10 \\\\")
    escribir_tex("tabla_p4_ljungbox.tex", filas_lb)

    figura_acfs([
        ("4.1 · Residuos de las indicadoras", "$e_t$ = venta menos el promedio de su día", r_e),
        ("4.2 · Residuos del AR(1)", f"$u_t = e_t - \\hat c - {phi:.3f}\\, e_{{t-1}}$", r_u),
        ("4.4 · Residuos de S1", "SARIMA(1,0,0)(1,0,0)$_6$ con constante, log", acf(res1, KMAX)),
        ("4.4 · Residuos de S2", "SARIMA(0,1,1)(0,1,1)$_6$, log, sin los 7 primeros", acf(res2, KMAX)),
    ], n)

    # 4.5 pronosticos a horizonte largo
    h = 60
    p = ajustes["S1"].params
    MEDIA_S1 = p[0] / ((1 - p[1]) * (1 - p[2]))
    fc1 = np.exp(ajustes["S1"].forecast(h))
    fc2 = np.exp(ajustes["S2"].forecast(h))
    figura_horizonte(ent.index, y, fc1, fc2, h)
    print(f"4.5 media de S1 en libras = {np.exp(MEDIA_S1):,.0f}")

    s1, s2 = ajustes["S1"].params, ajustes["S2"].params
    escribir_tex("datos_p4.tex", [
        macro("PhiResiduos", f"{phi:.3f}"),
        macro("FueraU", ", ".join(map(str, fuera_u))),
        macro("ReunoE", f"{r_e[0]:.3f}"),
        macro("ReseisE", f"{r_e[5]:.3f}"),
        macro("RedoceE", f"{r_e[11]:.3f}"),
        macro("RuseisU", f"{r_u[5]:.3f}"),
        macro("SunoConst", f"{s1[0]:.3f}"),
        macro("SunoPhi", f"{s1[1]:.3f}"),
        macro("SunoPhiS", f"{s1[2]:.3f}"),
        macro("SdosTheta", f"{s2[0]:.3f}"),
        macro("SdosThetaS", f"{s2[1]:.3f}"),
        macro("MediaSuno", libras(np.exp(MEDIA_S1))),
        macro("NombresSuno", nombres_s1.replace("_", r"\_")),
        macro("NombresSdos", nombres_s2.replace("_", r"\_")),
    ])


if __name__ == "__main__":
    main()

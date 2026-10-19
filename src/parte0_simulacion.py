"""Parte 0: procesos simulados.

Simula AR(1) con phi en {0, 0.5, 0.9, 1} y dos procesos sin memoria con media
cambiante (tendencia y patron semanal). Sobre 5000 replicas de cada uno mide
r_1, r_5, r_7, la tasa de rechazo de la prueba de retardo 1, el estimado OLS
de phi y la tasa de rechazo de Ljung-Box.

Uso:  python3 src/parte0_simulacion.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, CIRUELA, GRIS, Z95,
                   escribir_tex, macro, guardar, subtitulo, bandas_acf, piruleta)
from series import acf, ljung_box, ols_ar1

N = 300          # pasos que se conservan
QUEMA = 200      # pasos iniciales que se descartan
R = 5000         # replicas por escenario
KMAX = 10

# Una semilla distinta por escenario
SEMILLAS_AR = {0.0: 140405, 0.5: 77714, 0.9: 141903, 1.0: 1404057}
SEMILLA_TENDENCIA = 777141
SEMILLA_SEMANAL = 1419031


# --- Simulacion ---


def simular_ar1(phi, n, r, rng, c=0.0, sigma=1.0, quema=QUEMA):
    """r realizaciones de n pasos de un AR(1), una por fila, con Y_0 = 0."""
    eps = rng.normal(0.0, sigma, size=(r, n + quema))
    y = np.zeros((r, n + quema))
    previo = np.zeros(r)
    for t in range(n + quema):
        previo = c + phi * previo + eps[:, t]
        y[:, t] = previo
    return y[:, quema:]


def media_tendencia(n):
    t = np.arange(1, n + 1)
    return 2.0 * t / n


def media_semanal(n):
    t = np.arange(1, n + 1)
    return np.where(t % 7 == 6, -1.5, 0.25)


def simular_media(m, r, rng):
    """Y_t = m_t + eps_t, sin memoria."""
    return m + rng.normal(0.0, 1.0, size=(r, len(m)))


# --- Resumen de las replicas ---


def media_ee(x):
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def tasa_ee(rechazos):
    p = float(rechazos.mean())
    return p, math.sqrt(p * (1 - p) / len(rechazos))


def resumir(y, con_ols=True):
    r = acf(y, KMAX)
    banda = Z95 / math.sqrt(y.shape[1])
    res = {"r1": media_ee(r[:, 0]), "r5": media_ee(r[:, 4]), "r7": media_ee(r[:, 6]),
           "rech1": tasa_ee(np.abs(r[:, 0]) > banda)}
    if con_ols:
        res["phi"] = media_ee(ols_ar1(y)[1])
    for m in (6, 10):
        _, p = ljung_box(y, m)
        res[f"lb{m}"] = tasa_ee(p < 0.05)
    return res


def fmt(par, d=4):
    return f"${par[0]:.{d}f}$ & $({par[1]:.{d}f})$"


# --- Graficas ---


def figura_realizaciones(series):
    fig, ejes = plt.subplots(4, 1, figsize=(8.2, 7.4), sharex=True)
    colores = [PETROLEO, MOSTAZA, CIRUELA, TERRACOTA]
    t = np.arange(1, N + 1)
    for ax, (phi, y), color in zip(ejes, series.items(), colores):
        ax.fill_between(t, 0, y, color=color, alpha=0.18, lw=0)
        ax.plot(t, y, color=color, lw=0.9)
        ax.axhline(0, color=TINTA, lw=0.6)
        nombre = "caminata aleatoria" if phi == 1 else f"AR(1)"
        ax.set_title(fr"$\phi = {phi:g}$  ·  {nombre}")
        ax.set_ylabel("$y_t$")
    ejes[-1].set_xlabel("t")
    fig.tight_layout(h_pad=1.2)
    return guardar(fig, "p0_realizaciones.png")


def figura_media_cambiante(ejemplos, acfs):
    fig, ejes = plt.subplots(2, 3, figsize=(10, 5.2),
                             gridspec_kw={"height_ratios": [1.2, 1]})
    t = np.arange(1, N + 1)
    titulos = ["Ruido blanco", "Tendencia lineal", "Patrón semanal"]
    medias = [np.zeros(N), media_tendencia(N), media_semanal(N)]
    k = np.arange(1, KMAX + 1)
    for j, (y, m, r) in enumerate(zip(ejemplos, medias, acfs)):
        ax = ejes[0, j]
        if j == 2:
            # el patron semanal se ve mejor en las primeras cinco semanas
            t, y, m = t[:35], y[:35], m[:35]
            ax.plot(t, y, color=GRIS, lw=0.8, marker="o", ms=2.5)
            ax.step(t, m, where="mid", color=TERRACOTA, lw=1.8, label="$m_t$")
            texto = "primeras 5 semanas de una realización"
        else:
            ax.plot(t, y, color=GRIS, lw=0.6, alpha=0.9)
            ax.plot(t, m, color=TERRACOTA, lw=1.8, label="$m_t$")
            texto = "una realización y su media verdadera"
        ax.set_title(titulos[j])
        subtitulo(ax, texto)
        ax.set_xlabel("t")
        ax2 = ejes[1, j]
        banda = bandas_acf(ax2, N, KMAX)
        piruleta(ax2, k, r, banda)
        ax2.set_ylim(-0.25, 0.65)
        ax2.set_xticks(k)
        ax2.set_xlabel("rezago k")
        subtitulo(ax2, "promedio de $r_k$ en 5000 réplicas")
    ejes[0, 0].legend(loc="upper left")
    ejes[1, 0].set_ylabel("$r_k$")
    fig.tight_layout()
    return guardar(fig, "p0_media_cambiante.png")


# --- Principal ---


def main():
    print("Parte 0: simulando", R, "replicas por escenario...")

    # 0.2: AR(1)
    resultados, ejemplo = {}, {}
    replicas_ruido = None
    for phi, semilla in SEMILLAS_AR.items():
        rng = np.random.default_rng(semilla)
        y = simular_ar1(phi, N, R, rng)
        resultados[phi] = resumir(y)
        ejemplo[phi] = y[0]
        if phi == 0.0:
            replicas_ruido = y
        r = resultados[phi]
        print(f"  phi={phi:<4} r1={r['r1'][0]:+.4f}  r5={r['r5'][0]:+.4f} "
              f"(teo {phi**5:.4f})  rechazo={r['rech1'][0]:.4f}  "
              f"phi_ols={r['phi'][0]:.4f}")
    figura_realizaciones(ejemplo)

    filas = []
    for phi, r in resultados.items():
        filas.append(f"${phi:g}$ & {fmt(r['r1'])} & {fmt(r['r5'])} & ${phi**5:.4f}$ & "
                     f"{fmt(r['rech1'])} & {fmt(r['phi'])} \\\\")
    escribir_tex("tabla_p0_ar1.tex", filas)

    # 0.4: medias que cambian, sin memoria
    rng_t = np.random.default_rng(SEMILLA_TENDENCIA)
    rng_s = np.random.default_rng(SEMILLA_SEMANAL)
    y_t = simular_media(media_tendencia(N), R, rng_t)
    y_s = simular_media(media_semanal(N), R, rng_s)
    procesos = {"Ruido blanco ($\\phi=0$)": (replicas_ruido, resultados[0.0]),
                "Tendencia lineal": (y_t, resumir(y_t, con_ols=False)),
                "Patrón semanal": (y_s, resumir(y_s, con_ols=False))}
    filas, filas_lb = [], []
    for nombre, (y, r) in procesos.items():
        filas.append(f"{nombre} & {fmt(r['r1'])} & {fmt(r['r7'])} & {fmt(r['rech1'])} \\\\")
        filas_lb.append(f"{nombre} & {fmt(r['rech1'])} & {fmt(r['lb6'])} & {fmt(r['lb10'])} \\\\")
        print(f"  {nombre:<26} r1={r['r1'][0]:+.4f} r7={r['r7'][0]:+.4f} "
              f"rech1={r['rech1'][0]:.4f} LB6={r['lb6'][0]:.4f} LB10={r['lb10'][0]:.4f}")
    escribir_tex("tabla_p0_media.tex", filas)
    escribir_tex("tabla_p0_ljungbox.tex", filas_lb)

    acfs = [acf(y, KMAX).mean(axis=0) for y, _ in procesos.values()]
    ejemplos = [y[0] for y, _ in procesos.values()]
    figura_media_cambiante(ejemplos, acfs)

    # 0.5: verificacion de Ljung-Box contra statsmodels
    q_propio, p_propio = ljung_box(y_s[0], 10)
    sm = acorr_ljungbox(y_s[0], lags=[10])
    q_sm, p_sm = float(sm["lb_stat"].iloc[0]), float(sm["lb_pvalue"].iloc[0])
    print(f"  Ljung-Box propio Q={q_propio:.6f} p={p_propio:.6f} | "
          f"statsmodels Q={q_sm:.6f} p={p_sm:.6f}")

    r0, rt, rs = resultados[0.0], procesos["Tendencia lineal"][1], procesos["Patrón semanal"][1]
    escribir_tex("datos_p0.tex", [
        macro("BandaPcero", f"{Z95 / math.sqrt(N):.4f}"),
        macro("RechazoRuido", f"{r0['rech1'][0]:.4f}"),
        macro("RechazoRuidoEE", f"{r0['rech1'][1]:.4f}"),
        macro("RcincoCaminata", f"{resultados[1.0]['r5'][0]:.3f}"),
        macro("RunoCaminata", f"{resultados[1.0]['r1'][0]:.3f}"),
        macro("PhiOLSCaminata", f"{resultados[1.0]['phi'][0]:.4f}"),
        macro("PhiOLSnueve", f"{resultados[0.9]['phi'][0]:.4f}"),
        macro("PhiOLSmedio", f"{resultados[0.5]['phi'][0]:.4f}"),
        macro("PhiOLScero", f"{resultados[0.0]['phi'][0]:.4f}"),
        macro("RunoRuido", f"{r0['r1'][0]:.4f}"),
        macro("RechazoTend", f"{rt['rech1'][0]:.3f}"),
        macro("RechazoSem", f"{rs['rech1'][0]:.3f}"),
        macro("NoDetectaSem", f"{100 * (1 - rs['rech1'][0]):.0f}\\%"),
        macro("RunoTend", f"{rt['r1'][0]:.3f}"),
        macro("RsieteTend", f"{rt['r7'][0]:.3f}"),
        macro("RunoSem", f"{rs['r1'][0]:.3f}"),
        macro("RsieteSem", f"{rs['r7'][0]:.3f}"),
        macro("LBdiezRuido", f"{r0['lb10'][0]:.3f}"),
        macro("LBdiezTend", f"{rt['lb10'][0]:.3f}"),
        macro("LBdiezSem", f"{rs['lb10'][0]:.3f}"),
        macro("LBseisSem", f"{rs['lb6'][0]:.3f}"),
        macro("LBQpropio", f"{q_propio:.4f}"),
        macro("LBQsm", f"{q_sm:.4f}"),
        macro("LBPpropio", f"{p_propio:.4f}"),
    ])
    print("listo: figs/p0_*.png, doc/tabla_p0_*.tex, doc/datos_p0.tex")


if __name__ == "__main__":
    main()

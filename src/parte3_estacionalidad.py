"""Parte 3: tendencia, estacionalidad y autocorrelacion.

Descomposicion clasica multiplicativa a mano: media movil centrada de 7 dias
como tendencia, indices estacionales por dia de la semana y la ACF de los dias
abiertos del entrenamiento con Ljung-Box.

Uso:  python3 src/parte3_estacionalidad.py
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, GRIS, Z95,
                   escribir_tex, macro, libras, cientifica, guardar, subtitulo, bandas_acf, piruleta)
from datos import serie_trabajo, abiertos, FIN_ENTRENA, DIAS
from series import acf, ljung_box

KMAX = 21


def media_movil(s):
    """Media movil centrada de 7 dias con los dias cerrados como cero."""
    return s.fillna(0).rolling(7, center=True).mean()


def figura_tendencia(s, mm):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ab = s.dropna()
    ax.vlines(ab.index, 0, ab.values / 1000, color=PETROLEO, alpha=0.35, lw=1.4)
    ax.plot(mm.index, mm.values / 1000, color=TERRACOTA, lw=2.0,
            label="media móvil centrada de 7 días")
    ax.axvline(FIN_ENTRENA, color=TINTA, lw=0.8, ls="--")
    ax.text(FIN_ENTRENA, ax.get_ylim()[1] * 0.97, "  fin del entrenamiento",
            fontsize=7.5, color=TINTA, va="top")
    ax.set_ylabel("venta neta (miles de £)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_title("Venta neta diaria y su tendencia")
    subtitulo(ax, "barras: días abiertos · la media móvil usa los días cerrados como cero")
    ax.legend(loc="upper left")
    return guardar(fig, "p3_tendencia.png")


def figura_indices(indices, n_dias):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(indices))
    colores = [TERRACOTA if v == indices.max() else PETROLEO for v in indices.values]
    ax.bar(x, indices.values, color=colores, width=0.62)
    ax.axhline(1, color=TINTA, lw=0.8, ls="--")
    ax.axhline(indices.mean(), color=MOSTAZA, lw=1.2)
    ax.text(len(indices) - 0.5, indices.mean(), f" promedio {indices.mean():.3f}",
            color=MOSTAZA, fontsize=7.5, va="bottom", ha="right")
    for xi, v, n in zip(x, indices.values, n_dias):
        ax.text(xi, v - 0.04, f"{v:.2f}", ha="center", va="top", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(indices.index)
    ax.set_ylabel("índice estacional")
    ax.set_ylim(0, indices.max() * 1.18)
    ax.set_title("Índice estacional por día de la semana")
    subtitulo(ax, "promedio de $y_t / MA_t$ en los días abiertos del entrenamiento")
    return guardar(fig, "p3_indices.png")


def figura_acf(r, n, nombre, titulo, sub):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    k = np.arange(1, len(r) + 1)
    banda = bandas_acf(ax, n, len(r))
    piruleta(ax, k, r, banda)
    for j in (6, 12, 18):
        ax.axvline(j, color=MOSTAZA, lw=0.8, ls=":", zorder=0)
    ax.set_xticks(k)
    ax.set_xlabel("rezago k (días abiertos)")
    ax.set_ylabel("$r_k$")
    ax.set_title(titulo)
    subtitulo(ax, sub)
    return guardar(fig, nombre)


def main():
    s = serie_trabajo()
    mm = media_movil(s)
    figura_tendencia(s, mm)

    # 3.2 indices estacionales, solo entrenamiento
    ent = s[s.index <= FIN_ENTRENA]
    cociente = (ent / mm.loc[ent.index]).dropna()
    grupos = cociente.groupby(cociente.index.dayofweek)
    indices = grupos.mean()
    n_dias = grupos.size()
    indices.index = [DIAS[d] for d in indices.index]
    figura_indices(indices, n_dias)
    escribir_tex("tabla_p3_indices.tex",
                 [f"{d} & ${v:.3f}$ & ${n}$ \\\\" for (d, v), n in zip(indices.items(), n_dias)]
                 + ["\\midrule", f"Promedio & ${indices.mean():.3f}$ & \\\\"])
    print("indices:", indices.round(3).to_dict(), "promedio", round(indices.mean(), 3))

    # 3.3 ACF sobre los dias abiertos del entrenamiento, una fila por dia
    y = abiertos(ent).values
    n = len(y)
    r = acf(y, KMAX)
    banda = Z95 / np.sqrt(n)
    fuera = np.flatnonzero(np.abs(r) > banda) + 1
    pico = int(np.argmax(r) + 1)
    q6, p6 = ljung_box(y, 6)
    q12, p12 = ljung_box(y, 12)
    print(f"n={n} banda={banda:.4f} fuera={len(fuera)} {fuera.tolist()} pico={pico} "
          f"r1={r[0]:.3f} Q6={q6:.1f} p={p6:.2e} Q12={q12:.1f} p={p12:.2e}")
    figura_acf(r, n, "p3_acf.png", "ACF de la venta diaria (entrenamiento)",
               "días abiertos, una fila por día · banda: $\\pm 1.96/\\sqrt{n}$ · "
               "líneas punteadas en 6, 12 y 18")

    # Cuantos dias calendario hay entre dias abiertos separados 6 posiciones
    fechas = abiertos(ent).index
    salto = pd.Series((fechas[6:] - fechas[:-6]).days).value_counts()
    print("  dias calendario entre posiciones separadas por 6:", salto.to_dict())

    t_ini = mm.dropna()
    escribir_tex("datos_p3.tex", [
        macro("NEntrena", n),
        macro("BandaEntrena", f"{banda:.3f}"),
        macro("RezagosFuera", len(fuera)),
        macro("RunoReal", f"{r[0]:.3f}"),
        macro("RseisReal", f"{r[5]:.3f}"),
        macro("RdoceReal", f"{r[11]:.3f}"),
        macro("PicoACF", pico),
        macro("QseisReal", f"{q6:.1f}"),
        macro("QdoceReal", f"{q12:.1f}"),
        macro("PseisReal", cientifica(p6)),
        macro("PdoceReal", cientifica(p12)),
        macro("PromIndices", f"{indices.mean():.3f}"),
        macro("SaltoSeisFrac", f"{100 * salto.get(7, 0) / salto.sum():.0f}\\%"),
        macro("MMEnero", libras(mm["2011-01-10":"2011-08-31"].mean())),
        macro("MMNoviembre", libras(mm["2011-11-01":"2011-11-30"].mean())),
        macro("MMMax", libras(t_ini.max())),
    ])


if __name__ == "__main__":
    main()

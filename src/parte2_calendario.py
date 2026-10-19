"""Parte 2: el calendario.

Construye la venta neta diaria, clasifica los dias sin transacciones contra el
calendario de bank holidays del Reino Unido y revisa el ultimo dia.

Uso:  python3 src/parte2_calendario.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, CREMA, GRIS,
                   escribir_tex, macro, miles, libras, guardar, subtitulo)
from datos import datos_limpios, serie_diaria, DIAS

# Bank holidays de Inglaterra y Gales dentro del periodo (gov.uk)
FERIADOS = {
    "2010-12-27": ("Navidad (día sustituto)", False),
    "2010-12-28": ("Boxing Day (día sustituto)", False),
    "2011-01-03": ("Año Nuevo (día sustituto)", False),
    "2011-04-22": ("Viernes Santo", True),
    "2011-04-25": ("Lunes de Pascua", True),
    "2011-04-29": ("Boda real (solo en 2011)", False),
    "2011-05-02": ("Early May bank holiday", False),
    "2011-05-30": ("Spring bank holiday", False),
    "2011-08-29": ("Summer bank holiday", False),
}
OTROS = {
    "2011-04-24": ("Domingo de Pascua (no es bank holiday)", True),
}


def clasificar(d):
    k = d.strftime("%Y-%m-%d")
    if k in FERIADOS:
        return FERIADOS[k]
    if k in OTROS:
        return OTROS[k]
    if pd.Timestamp("2010-12-23") < d < pd.Timestamp("2011-01-04"):
        return ("Cierre de fin de año", False)
    if d.dayofweek == 5:
        return ("Sábado", False)
    return ("Sin explicar", False)


def figura_calendario(s):
    """Mapa de calor tipo calendario: una columna por semana, una fila por dia."""
    inicio = s.index[0] - pd.Timedelta(days=s.index[0].dayofweek)
    semanas = ((s.index - inicio).days // 7).values
    filas = s.index.dayofweek.values
    nsem = semanas.max() + 1
    m = np.full((7, nsem), np.nan)
    m[filas, semanas] = s.values / 1000

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "venta", ["#F3E9D2", "#9FC9B9", PETROLEO, TINTA])
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.set_facecolor("white")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    im = ax.pcolormesh(np.arange(nsem + 1), np.arange(8), np.ma.masked_invalid(m),
                       cmap=cmap, edgecolors="white", linewidth=1.2)

    for d, v in s.items():
        if np.isnan(v):
            x, y = (d - inicio).days // 7, d.dayofweek
            nombre, _ = clasificar(d)
            color = "#E4E1DB" if nombre == "Sábado" else TERRACOTA
            ax.add_patch(plt.Rectangle((x + 0.08, y + 0.08), 0.84, 0.84,
                                       facecolor=color, edgecolor="none"))
    ult = s.index[-1]
    ax.add_patch(plt.Rectangle(((ult - inicio).days // 7, ult.dayofweek), 1, 1,
                               fill=False, edgecolor=MOSTAZA, lw=2.2))

    ax.set_yticks(np.arange(7) + 0.5)
    ax.set_yticklabels([d[:3] for d in DIAS])
    ax.invert_yaxis()
    meses = pd.date_range(s.index[0], s.index[-1], freq="MS")
    ax.set_xticks([(m_ - inicio).days // 7 + 0.5 for m_ in meses])
    ax.set_xticklabels([m_.strftime("%b %y") for m_ in meses])
    ax.tick_params(length=0)
    ax.set_title("Venta neta diaria sobre el calendario completo")
    subtitulo(ax, "gris: sábado cerrado · terracota: feriado o cierre de fin de año · "
                  "borde mostaza: último día (incompleto)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("miles de £", fontsize=8)
    cb.outline.set_visible(False)
    return guardar(fig, "p2_calendario.png")


def main():
    limpio = datos_limpios()
    s = serie_diaria(limpio)

    # 2.1
    n_cal = len(s)
    n_abiertos = int(s.notna().sum())
    print(f"dias calendario={n_cal}  con transacciones={n_abiertos}  "
          f"desde {s.index[0]:%Y-%m-%d} hasta {s.index[-1]:%Y-%m-%d}")

    # 2.2
    cerrados = s[s.isna()].index
    clases = pd.Series([clasificar(d)[0] for d in cerrados], index=cerrados)
    sabados = int((clases == "Sábado").sum())
    print(f"  sabados cerrados={sabados}  dias de la semana cerrados: "
          f"{dict(cerrados.dayofweek.value_counts())}")
    sabados_abiertos = int(s[s.index.dayofweek == 5].notna().sum())
    lineas = []
    for d in cerrados:
        nombre, movil = clasificar(d)
        if nombre == "Sábado":
            continue
        lineas.append(f"{d:%d/%m/%Y} & {DIAS[d.dayofweek]} & {nombre} & "
                      f"{'sí' if movil else 'no'} \\\\")
        print(f"  {d:%a %Y-%m-%d}  {nombre}")
    sin_explicar = int((clases == "Sin explicar").sum())
    escribir_tex("tabla_p2_cerrados.tex", lineas)

    # 2.4: el ultimo dia
    ultimo = s.index[-1]
    filas_ult = limpio[limpio["InvoiceDate"].dt.normalize() == ultimo]
    hora_fin = filas_ult["InvoiceDate"].max()
    cierre = limpio.groupby(limpio["InvoiceDate"].dt.normalize())["InvoiceDate"].max()
    cierre_h = (cierre.dt.hour + cierre.dt.minute / 60)
    misma = s[(s.index.dayofweek == ultimo.dayofweek) & s.notna()].iloc[-9:-1]
    print(f"  ultimo dia {ultimo:%Y-%m-%d}: hasta {hora_fin:%H:%M}, venta {s.iloc[-1]:,.0f}; "
          f"mediana de la hora de la ultima factura = {cierre_h.median():.2f}; "
          f"promedio de los 8 viernes anteriores = {misma.mean():,.0f}")

    figura_calendario(s)

    escribir_tex("datos_p2.tex", [
        macro("DiasCalendario", n_cal),
        macro("DiasAbiertos", n_abiertos),
        macro("DiasCerrados", n_cal - n_abiertos),
        macro("SabadosCerrados", sabados),
        macro("SabadosAbiertos", sabados_abiertos),
        macro("CerradosNoSabado", n_cal - n_abiertos - sabados),
        macro("SinExplicar", sin_explicar),
        macro("UltimaHora", f"{hora_fin:%H:%M}"),
        macro("VentaUltimoDia", libras(s.iloc[-1])),
        macro("FilasUltimoDia", miles(len(filas_ult))),
        macro("HoraCierreMediana", f"{int(cierre_h.median())}:{int(round(cierre_h.median() % 1 * 60)):02d}"),
        macro("VentaViernesPrevios", libras(misma.mean())),
    ])


if __name__ == "__main__":
    main()

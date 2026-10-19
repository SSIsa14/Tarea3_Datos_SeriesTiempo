"""Parte 6: detras de la serie hay clientes, analisis RFM.

Calcula recencia, frecuencia y monto al 1 de octubre de 2011 con los datos
limpios que tienen CustomerID, asigna puntajes por quintiles (con el
desempate al azar del enunciado) y mide que tan bien anticipan quien compra
entre el 1 de octubre y el 8 de diciembre.

Uso:  python3 src/parte6_rfm.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, CIRUELA, GRIS,
                   escribir_tex, macro, miles, libras, pct, guardar, subtitulo)
from datos import datos_limpios, FIN_PRUEBA

CORTE = pd.Timestamp("2011-10-01")
FIN = FIN_PRUEBA + pd.Timedelta(days=1)   # hasta el 8 de diciembre inclusive
SEMILLA = 2026                            # la fija el enunciado


def calcular_rfm(df):
    antes = df[df["InvoiceDate"] < CORTE]
    ventas = antes[~antes["InvoiceNo"].str.startswith("C")]
    rfm = pd.DataFrame({
        "R": (CORTE - ventas.groupby("CustomerID")["InvoiceDate"].max().dt.normalize()).dt.days,
        "F": ventas.groupby("CustomerID")["InvoiceNo"].nunique(),
    })
    rfm["M"] = antes.groupby("CustomerID")["Monto"].sum().reindex(rfm.index)
    solo_cancel = antes["CustomerID"].nunique() - len(rfm)
    return rfm.sort_index(), solo_cancel


def puntajes(rfm):
    """Quintiles 1-5. Se baraja con la semilla 2026 y se rompe el empate por orden."""
    rng = np.random.default_rng(SEMILLA)
    barajado = rfm.iloc[rng.permutation(len(rfm))]
    etiquetas = [1, 2, 3, 4, 5]
    p = pd.DataFrame(index=barajado.index)
    p["R_s"] = pd.qcut((-barajado["R"]).rank(method="first"), 5, labels=etiquetas).astype(int)
    p["F_s"] = pd.qcut(barajado["F"].rank(method="first"), 5, labels=etiquetas).astype(int)
    p["M_s"] = pd.qcut(barajado["M"].rank(method="first"), 5, labels=etiquetas).astype(int)
    return p.loc[rfm.index]


def figura_tasas(tasas):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colores = {"R": TERRACOTA, "F": PETROLEO, "M": MOSTAZA}
    x = np.arange(1, 6)
    for dim, serie in tasas.items():
        ax.plot(x, serie.values, color=colores[dim], lw=2, marker="o", ms=6,
                markeredgecolor="white")
        ax.text(5.12, serie.values[-1], dim, color=colores[dim], fontweight="bold",
                va="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("puntaje (5 = mejor)")
    ax.set_ylabel("compró entre oct. y dic.")
    ax.set_title("Fracción de clientes que vuelve a comprar, por puntaje")
    subtitulo(ax, "clientes con historia antes del 1/10/2011")
    return guardar(fig, "p6_tasas.png")


def figura_reparto(partes):
    fig, ax = plt.subplots(figsize=(8.5, 1.9))
    colores = [TERRACOTA, PETROLEO, MOSTAZA]
    izq = 0
    for (nombre, frac), color in zip(partes.items(), colores):
        ax.barh(0, frac, left=izq, color=color, height=0.55)
        ax.text(izq + frac / 2, 0, f"{nombre}\n{frac:.0%}", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        izq += frac
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("¿Quién generó la venta neta del 1/10 al 8/12/2011?")
    subtitulo(ax, "clientes con CustomerID, incluidos los nuevos")
    return guardar(fig, "p6_reparto.png")


def main():
    df = datos_limpios()
    df = df[df["CustomerID"].notna()]

    # 6.1
    rfm, solo_cancel = calcular_rfm(df)
    n = len(rfm)
    m_neg = int((rfm["M"] <= 0).sum())
    print(f"clientes={n}  con M<=0={m_neg}  solo cancelaciones (sin R)={solo_cancel}")

    # 6.2
    p = puntajes(rfm)
    rfm = rfm.join(p)
    frac_f1 = (rfm["F"] == 1).mean()
    try:
        pd.qcut(rfm["F"], 5)
        problema = "ninguno"
    except ValueError as e:
        problema = str(e).split(".")[0]
    print(f"  F=1 en {frac_f1:.3f} de los clientes; qcut directo sobre F: {problema}")
    rangos_f = rfm.groupby("F_s")["F"].agg(["min", "max"])
    print("  rangos de F por puntaje:", rangos_f.to_dict("index"))

    # 6.3
    despues = df[(df["InvoiceDate"] >= CORTE) & (df["InvoiceDate"] < FIN)]
    vent_desp = despues[~despues["InvoiceNo"].str.startswith("C")]
    compradores = set(vent_desp["CustomerID"])
    rfm["compra"] = rfm.index.isin(compradores).astype(int)
    tasas = {d: rfm.groupby(f"{d}_s")["compra"].mean() for d in ("R", "F", "M")}
    figura_tasas(tasas)
    rfm["suma"] = rfm["R_s"] + rfm["F_s"] + rfm["M_s"]
    auc = {
        "R": roc_auc_score(rfm["compra"], -rfm["R"]),
        "F": roc_auc_score(rfm["compra"], rfm["F"]),
        "M": roc_auc_score(rfm["compra"], rfm["M"]),
        "suma": roc_auc_score(rfm["compra"], rfm["suma"]),
    }
    print("  tasas:", {d: t.round(3).tolist() for d, t in tasas.items()})
    print("  AUC:", {k: round(v, 3) for k, v in auc.items()})
    filas = [f"{k} & " + " & ".join(f"${100 * tasas[d].loc[k]:.1f}\\%$" for d in ("R", "F", "M"))
             + " \\\\" for k in range(1, 6)]
    escribir_tex("tabla_p6_tasas.tex", filas)
    escribir_tex("tabla_p6_auc.tex", [
        f"AUC & ${auc['R']:.3f}$ & ${auc['F']:.3f}$ & ${auc['M']:.3f}$ & ${auc['suma']:.3f}$ \\\\"])

    # 6.4 campeones
    campeon = (rfm["R_s"] >= 4) & (rfm["F_s"] >= 4) & (rfm["M_s"] >= 4)
    venta_cli = despues.groupby("CustomerID")["Monto"].sum()
    total = venta_cli.sum()
    venta_camp = venta_cli.reindex(rfm.index[campeon]).fillna(0).sum()

    # 6.5 clientes nuevos: sin ninguna factura de venta antes del 1 de octubre
    nuevos = venta_cli.index.difference(rfm.index)
    venta_nuevos = venta_cli.loc[nuevos].sum()
    n_nuevos_compra = len(set(vent_desp["CustomerID"]) - set(rfm.index))
    print(f"  campeones={campeon.sum()} ({campeon.mean():.3f}) venta={venta_camp / total:.3f}  "
          f"nuevos={n_nuevos_compra} venta={venta_nuevos / total:.3f}  total={total:,.0f}")
    figura_reparto({"Campeones": venta_camp / total,
                    "Otros clientes con historia": (total - venta_camp - venta_nuevos) / total,
                    "Nuevos": venta_nuevos / total})

    escribir_tex("datos_p6.tex", [
        macro("ClientesRFM", miles(n)),
        macro("ClientesMneg", miles(m_neg)),
        macro("ClientesSoloCancel", miles(solo_cancel)),
        macro("FracFuno", pct(frac_f1)),
        macro("FmaxUno", int(rangos_f.loc[1, "max"])),
        macro("FmaxDos", int(rangos_f.loc[2, "max"])),
        macro("FminTres", int(rangos_f.loc[3, "min"])),
        macro("FracCompra", pct(rfm["compra"].mean())),
        macro("AUCR", f"{auc['R']:.3f}"),
        macro("AUCF", f"{auc['F']:.3f}"),
        macro("AUCM", f"{auc['M']:.3f}"),
        macro("AUCSuma", f"{auc['suma']:.3f}"),
        macro("NCampeones", miles(campeon.sum())),
        macro("FracCampeones", pct(campeon.mean())),
        macro("VentaCampeones", pct(venta_camp / total)),
        macro("NNuevos", miles(n_nuevos_compra)),
        macro("VentaNuevos", pct(venta_nuevos / total)),
        macro("VentaOctDic", libras(total)),
    ])


if __name__ == "__main__":
    main()

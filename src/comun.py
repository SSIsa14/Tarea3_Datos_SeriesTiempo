"""Utilidades comunes: rutas, salida en LaTeX y el estilo de las graficas."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIGS = os.path.join(RAIZ, "figs")
DOC = os.path.join(RAIZ, "doc")
DATA = os.path.join(RAIZ, "data")

Z95 = 1.959964  # cuantil 0.975 de la normal estandar


# --- Estilo de las graficas ---
# Paleta propia: tinta, terracota, verde petroleo, mostaza y ciruela sobre un
# fondo crema muy suave.

TINTA = "#1F2A36"
TERRACOTA = "#C8553D"
PETROLEO = "#2A7F7A"
MOSTAZA = "#D99A1E"
CIRUELA = "#6B4E71"
GRIS = "#8C8C8C"
CREMA = "#FBF8F3"
PALETA = [PETROLEO, TERRACOTA, MOSTAZA, CIRUELA, TINTA]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": CREMA,
    "axes.edgecolor": "#B8B2A7",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#D9D3C7",
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.titlecolor": TINTA,
    "axes.labelcolor": TINTA,
    "axes.labelsize": 9,
    "axes.prop_cycle": matplotlib.cycler(color=PALETA),
    "xtick.color": "#5A5A5A",
    "ytick.color": "#5A5A5A",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.size": 9,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def subtitulo(ax, texto):
    """Linea gris debajo del titulo, para decir que se esta viendo."""
    ax.text(0, 1.015, texto, transform=ax.transAxes, fontsize=7.5,
            color=GRIS, va="bottom", ha="left")
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=14)


def guardar(fig, nombre):
    ruta = os.path.join(FIGS, nombre)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def bandas_acf(ax, n, kmax):
    """Banda sombreada de +-1.96/sqrt(n) para las graficas de ACF."""
    b = Z95 / n ** 0.5
    ax.axhspan(-b, b, color=PETROLEO, alpha=0.12, lw=0)
    ax.axhline(0, color=TINTA, lw=0.7)
    ax.set_xlim(0.3, kmax + 0.7)
    return b


def piruleta(ax, k, r, banda, color=TINTA):
    """ACF como piruletas: las que se salen de la banda van en terracota."""
    fuera = abs(r) > banda
    ax.vlines(k, 0, r, color=color, lw=1.1)
    ax.scatter(k[~fuera], r[~fuera], s=16, color=color, zorder=3)
    ax.scatter(k[fuera], r[fuera], s=22, color=TERRACOTA, zorder=4,
               edgecolor="white", linewidth=0.6)


# --- Salida en LaTeX ---

CABECERA = ("% Generado por los scripts de src/.\n"
            "% No editar a mano: se sobrescribe en cada ejecucion.\n")


def escribir_tex(nombre, lineas):
    ruta = os.path.join(DOC, nombre)
    with open(ruta, "w") as f:
        f.write(CABECERA + "\n".join(lineas) + "\n")
    return ruta


def macro(nombre, valor):
    return f"\\newcommand{{\\{nombre}}}{{{valor}}}"


def miles(x, decimales=0):
    return f"{x:,.{decimales}f}".replace(",", "{,}")


def libras(x, decimales=0):
    signo = "-" if x < 0 else ""
    return f"{signo}\\pounds{miles(abs(x), decimales)}"


def pct(x, decimales=1):
    return f"{100 * x:.{decimales}f}\\%"

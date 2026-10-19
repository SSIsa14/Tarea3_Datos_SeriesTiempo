"""Parte 5: pronostico y fuga.

Seis modelos (B1, B2, M1, M2, S1, S2) evaluados de tres formas: validacion
cruzada aleatoria de 5 folds, origen movil semanal dentro del entrenamiento
y origen movil semanal en el periodo de prueba. Al final, la trampa de usar
la media movil centrada como predictor.

Uso:  python3 src/parte5_pronostico.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import KFold

from comun import (TINTA, TERRACOTA, PETROLEO, MOSTAZA, CIRUELA, GRIS,
                   escribir_tex, macro, miles, libras, pct, guardar, subtitulo)
from datos import (serie_trabajo, abiertos, FIN_ENTRENA, PRIMER_ORIGEN_PRUEBA,
                   INICIO_PRUEBA, FIN_PRUEBA)
from parte3_estacionalidad import media_movil
from parte4_sarima import ajustar, indicadoras

SEMILLAS_CV = [141903 + i for i in range(10)]
LETRAS = {"B1": "Buno", "B2": "Bdos", "M1": "Muno", "M2": "Mdos", "S1": "Suno", "S2": "Sdos"}
MODELOS = ["B1", "B2", "M1", "M2", "S1", "S2"]
ORIGENES_B = pd.date_range("2011-05-01", "2011-09-30", freq="W-SUN")
ORIGENES_C = pd.date_range(PRIMER_ORIGEN_PRUEBA, FIN_PRUEBA, freq="W-SUN")
T0 = pd.Timestamp("2010-12-01")


# --- Predictores ---


def rezagos(y):
    """y_{t-7} e y_{t-14}: el mismo dia de la semana abierto mas reciente y el
    anterior a ese. Si la semana pasada fue feriado, se salta a la de antes."""
    df = pd.DataFrame({"y": y})
    grupo = df.groupby(y.index.dayofweek)["y"]
    df["lag7"] = grupo.shift(1)
    df["lag14"] = grupo.shift(2)
    return df[["lag7", "lag14"]]


def disenio(fechas, modelo, lags, extra=None):
    X = [indicadoras(fechas)]
    if modelo == "M1":
        X.append(((fechas - T0).days.values.astype(float))[:, None])
    if modelo in ("M2", "M2+MA"):
        X.append(lags.loc[fechas].values)
    if modelo == "M2+MA":
        X.append(extra.loc[fechas].values[:, None])
    return np.column_stack(X)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def ajustar_ols(fechas, y, modelo, lags, extra=None):
    """Ajusta M1, M2 o M2+MA quitando las filas que no tienen rezago."""
    X = disenio(fechas, modelo, lags, extra)
    ok = ~np.isnan(X).any(axis=1)
    return ols(X[ok], y[ok])


def pronosticar(modelo, hist, objetivos, lags, extra=None, conv=None):
    """Pronostico de `modelo` para las fechas `objetivos` con la historia `hist`
    (dias abiertos hasta el origen, inclusive)."""
    if modelo == "B1":
        return np.full(len(objetivos), hist.mean())
    if modelo == "B2":
        ult = hist.groupby(hist.index.dayofweek).last()
        return ult.loc[objetivos.dayofweek].values
    if modelo in ("M1", "M2", "M2+MA"):
        beta = ajustar_ols(hist.index, hist.values, modelo, lags, extra)
        return disenio(objetivos, modelo, lags, extra) @ beta
    # S1 y S2: el horizonte es la posicion entre los dias abiertos tras el origen
    res = ajustar(np.log(hist.values), modelo)
    if conv is not None:
        conv.append(bool(res.mle_retvals["converged"]))
    return np.exp(res.forecast(len(objetivos)))


# --- Las tres formas de evaluar ---


def metricas(err):
    err = np.asarray(err)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def cv_aleatoria(ent, modelo, lags, extra=None):
    """5 folds al azar sobre los dias abiertos del entrenamiento, 10 semillas."""
    fechas = ent.index
    if modelo in ("M2", "M2+MA"):
        X = disenio(fechas, modelo, lags, extra)
        fechas = fechas[~np.isnan(X).any(axis=1)]
    y = ent.loc[fechas]
    maes, rmses = [], []
    for semilla in SEMILLAS_CV:
        err = []
        for i_tr, i_te in KFold(5, shuffle=True, random_state=semilla).split(fechas):
            tr, te = y.iloc[i_tr], y.iloc[i_te]
            if modelo == "B1":
                pred = np.full(len(te), tr.mean())
            else:
                beta = ajustar_ols(tr.index, tr.values, modelo, lags, extra)
                pred = disenio(te.index, modelo, lags, extra) @ beta
            err.extend(te.values - pred)
        mae, rmse = metricas(err)
        maes.append(mae)
        rmses.append(rmse)
    return (np.mean(maes), np.std(maes, ddof=1)), (np.mean(rmses), np.std(rmses, ddof=1))


def origen_movil(ab, origenes, fin, modelo, lags, extra=None, conv=None):
    """Cada domingo pronostica los dias abiertos de lunes a domingo siguientes."""
    salida = []
    for o in origenes:
        hist = ab[ab.index <= o]
        objetivos = ab.index[(ab.index > o) & (ab.index <= o + pd.Timedelta(days=7))
                             & (ab.index <= fin)]
        if len(objetivos) == 0:
            continue
        pred = pronosticar(modelo, hist, objetivos, lags, extra, conv)
        salida.append(pd.DataFrame({"real": ab.loc[objetivos].values, "pred": pred},
                                   index=objetivos))
    return pd.concat(salida)


# --- Graficas ---


def figura_prueba(ab, preds, metr):
    fig, ejes = plt.subplots(3, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    colores = {"B1": GRIS, "B2": MOSTAZA, "M1": CIRUELA, "M2": PETROLEO,
               "S1": "#4F7CAC", "S2": TERRACOTA}
    real = ab[(ab.index >= INICIO_PRUEBA - pd.Timedelta(days=14)) & (ab.index <= FIN_PRUEBA)]
    ganador = min(MODELOS, key=lambda m: metr[m][0])
    for ax, m in zip(ejes.flat, MODELOS):
        ax.plot(real.index, real.values / 1000, color="#B9B2A6", lw=1.0, marker="o",
                ms=2.4, label="real")
        p = preds[m]["pred"]
        ax.plot(p.index, p.values / 1000, color=colores[m], lw=1.7, label="pronóstico")
        ax.axvline(INICIO_PRUEBA, color=TINTA, lw=0.6, ls="--")
        marca = "  ★ mejor MAE" if m == ganador else ""
        ax.set_title(f"{m}{marca}")
        subtitulo(ax, f"MAE £{metr[m][0]:,.0f} · RMSE £{metr[m][1]:,.0f}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for ax in ejes[:, 0]:
        ax.set_ylabel("miles de £")
    ejes[0, 0].legend(loc="upper left")
    fig.tight_layout(h_pad=1.4)
    return guardar(fig, "p5_prueba.png")


# --- Principal ---


def main():
    s = serie_trabajo()
    ab = abiertos(s)
    ent = ab[ab.index <= FIN_ENTRENA]
    lags = rezagos(ab)
    mm = media_movil(s)

    print("9 de mayo:", lags.loc["2011-05-09"].to_dict(),
          "| 18 de abril vale", round(ab.loc["2011-04-18"], 2))

    res = {}
    preds_c = {}
    conv = {"S1": [], "S2": []}
    for m in MODELOS:
        fila = {}
        if m in ("B1", "M1", "M2"):
            fila["a"] = cv_aleatoria(ent, m, lags)
        b = origen_movil(ab, ORIGENES_B, FIN_ENTRENA, m, lags, conv=conv.get(m))
        c = origen_movil(ab, ORIGENES_C, FIN_PRUEBA, m, lags, conv=conv.get(m))
        fila["b"] = metricas(b["real"] - b["pred"])
        fila["c"] = metricas(c["real"] - c["pred"])
        fila["nb"], fila["nc"] = len(b), len(c)
        res[m] = fila
        preds_c[m] = c
        a = fila.get("a")
        print(f"{m}: (a) {'' if a is None else f'MAE {a[0][0]:,.0f}±{a[0][1]:,.0f} RMSE {a[1][0]:,.0f}±{a[1][1]:,.0f}'}"
              f"  (b) MAE {fila['b'][0]:,.0f} RMSE {fila['b'][1]:,.0f}"
              f"  (c) MAE {fila['c'][0]:,.0f} RMSE {fila['c'][1]:,.0f}  n_b={len(b)} n_c={len(c)}")
    for m, lista in conv.items():
        print(f"  {m}: convergio en {sum(lista)} de {len(lista)} origenes")

    # 5.4 la trampa
    trampa = {"a": cv_aleatoria(ent, "M2+MA", lags, mm)}
    b = origen_movil(ab, ORIGENES_B, FIN_ENTRENA, "M2+MA", lags, mm)
    trampa["b"] = metricas(b["real"] - b["pred"])
    print(f"M2+MA: (a) MAE {trampa['a'][0][0]:,.0f} RMSE {trampa['a'][1][0]:,.0f}  "
          f"(b) MAE {trampa['b'][0]:,.0f} RMSE {trampa['b'][1]:,.0f}")

    # Tabla unica
    def celda_a(a):
        if a is None:
            return "\\multicolumn{2}{c|}{---}"
        return (f"${miles(a[0][0])} \\pm {miles(a[0][1])}$ & "
                f"${miles(a[1][0])} \\pm {miles(a[1][1])}$")

    filas = []
    for m in MODELOS + ["M2+MA"]:
        f = res[m] if m in res else trampa
        if m == "M2+MA":
            filas.append("\\midrule")
            c_txt = "\\multicolumn{2}{c|}{sólo con trampa}"
            nombre = "M2 + MA (5.4)"
        else:
            c_txt = f"${miles(f['c'][0])}$ & ${miles(f['c'][1])}$"
            nombre = m
        filas.append(f"{nombre} & {celda_a(f.get('a'))} & ${miles(f['b'][0])}$ & "
                     f"${miles(f['b'][1])}$ & {c_txt} \\\\")
    escribir_tex("tabla_p5_errores.tex", filas)

    metr_c = {m: res[m]["c"] for m in MODELOS}
    figura_prueba(ab, preds_c, metr_c)
    ganador = min(MODELOS, key=lambda m: metr_c[m][0])

    prueba = ab[(ab.index >= INICIO_PRUEBA) & (ab.index <= FIN_PRUEBA)]
    escribir_tex("datos_p5.tex", [
        macro("NdiasB", res["B1"]["nb"]),
        macro("NdiasC", res["B1"]["nc"]),
        macro("NdiasPrueba", len(prueba)),
        macro("NorigenesB", len(ORIGENES_B)),
        macro("NorigenesC", len(ORIGENES_C)),
        macro("ConvSuno", f"{sum(conv['S1'])} de {len(conv['S1'])}"),
        macro("ConvSdos", f"{sum(conv['S2'])} de {len(conv['S2'])}"),
        macro("Ganador", ganador),
        macro("MediaEntrena", libras(ent.mean())),
        macro("MediaPrueba", libras(prueba.mean())),
        macro("MediaMayoSep", libras(ent["2011-05-01":].mean())),
        macro("LagMayo", libras(lags.loc["2011-05-09", "lag7"], 2)),
    ] + [macro(f"MAEc{LETRAS[m]}", miles(res[m]["c"][0])) for m in MODELOS]
      + [macro(f"MAEb{LETRAS[m]}", miles(res[m]["b"][0])) for m in MODELOS]
      + [macro(f"MAEa{LETRAS[m]}", miles(res[m]["a"][0][0])) for m in ("B1", "M1", "M2")]
      + [macro("MAEaTrampa", miles(trampa["a"][0][0])),
         macro("MAEbTrampa", miles(trampa["b"][0]))])


if __name__ == "__main__":
    main()

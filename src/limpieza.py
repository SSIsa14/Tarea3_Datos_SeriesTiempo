"""Carga y limpieza de Online Retail.

La funcion limpiar(df) aplica, en orden, las reglas que se justifican en la
Parte 1 del documento. La bitacora (cuantas filas y cuantas libras elimina
cada regla) sale de la misma lista de reglas, asi que no se puede desfasar
de lo que de verdad hace la limpieza.
"""

import os

import pandas as pd

from comun import DATA

XLSX = os.path.join(DATA, "Online Retail.xlsx")
CACHE = os.path.join(DATA, "online_retail.pkl")

# Codigos que no son venta de mercancia: envio, comisiones y muestras.
# M (Manual) y D (Discount) se quedan: corrigen el precio de ventas reales.
# Los DCGS..., gift_... y PADS si son productos.
NO_PRODUCTO = {"POST", "DOT", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK", "S"}


def cargar():
    """Lee el .xlsx original. InvoiceNo, StockCode y CustomerID van como texto.

    La primera lectura del Excel tarda unos 20 segundos; se guarda una copia en
    data/online_retail.pkl para que los demas scripts no la repitan.
    """
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) > os.path.getmtime(XLSX):
        return pd.read_pickle(CACHE)
    df = pd.read_excel(XLSX, dtype={"InvoiceNo": str, "StockCode": str,
                                    "CustomerID": str})
    df.to_pickle(CACHE)
    return df


# --- Reglas, en el orden en que se aplican ---


def _duplicadas(df):
    return df.duplicated(keep="first")


def _prefijo_a(df):
    return df["InvoiceNo"].str.startswith("A")


def _no_producto(df):
    return df["StockCode"].isin(NO_PRODUCTO)


def _precio_no_positivo(df):
    return df["UnitPrice"] <= 0


def _negativas_sin_c(df):
    return (df["Quantity"] < 0) & ~df["InvoiceNo"].str.startswith("C")


REGLAS = [
    ("Filas duplicadas exactas (se conserva la primera)", _duplicadas),
    ("Facturas con prefijo A (ajuste de deuda incobrable)", _prefijo_a),
    ("Códigos que no son mercancía (envío, comisiones, muestras)", _no_producto),
    ("Precio unitario cero o negativo", _precio_no_positivo),
    ("Cantidad negativa en facturas que no son cancelación", _negativas_sin_c),
]


def limpiar(df, bitacora=None):
    """Devuelve el DataFrame limpio, con una columna Monto = Quantity * UnitPrice.

    Las cancelaciones (prefijo C) se quedan con su monto negativo, para que
    resten el dia en que ocurren. Las filas sin CustomerID tambien se quedan:
    son ventas reales; la Parte 6 las filtra porque ahi si importa quien compro.
    Si se pasa una lista en `bitacora`, se le agrega una fila por regla.
    """
    df = df.copy()
    df["Monto"] = df["Quantity"] * df["UnitPrice"]
    for nombre, regla in REGLAS:
        quitar = regla(df)
        if bitacora is not None:
            bitacora.append({"regla": nombre, "filas": int(quitar.sum()),
                             "monto": float(df.loc[quitar, "Monto"].sum())})
        df = df.loc[~quitar]
    return df.reset_index(drop=True)


def serie_diaria(limpio):
    """Venta neta diaria sobre el calendario completo (dias sin datos = NaN)."""
    dia = limpio["InvoiceDate"].dt.normalize()
    neta = limpio.groupby(dia)["Monto"].sum()
    calendario = pd.date_range(neta.index.min(), neta.index.max(), freq="D")
    return neta.reindex(calendario)

"""La serie de trabajo: venta neta diaria y la particion entrenamiento/prueba."""

import pandas as pd

from limpieza import cargar, limpiar

FIN_ENTRENA = pd.Timestamp("2011-09-30")
PRIMER_ORIGEN_PRUEBA = pd.Timestamp("2011-10-02")   # domingo: historia, no se evalua
INICIO_PRUEBA = pd.Timestamp("2011-10-03")
FIN_PRUEBA = pd.Timestamp("2011-12-08")

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

_limpio = None


def datos_limpios():
    global _limpio
    if _limpio is None:
        _limpio = limpiar(cargar())
    return _limpio


def serie_diaria(limpio=None):
    """Venta neta diaria sobre el calendario completo (dias sin datos = NaN)."""
    if limpio is None:
        limpio = datos_limpios()
    dia = limpio["InvoiceDate"].dt.normalize()
    neta = limpio.groupby(dia)["Monto"].sum()
    calendario = pd.date_range(neta.index.min(), neta.index.max(), freq="D")
    return neta.reindex(calendario)


def serie_trabajo():
    """La serie de la Parte 2 en adelante: calendario completo, dias cerrados
    como NaN y sin el ultimo dia (9 de diciembre), que esta incompleto."""
    s = serie_diaria()
    return s.iloc[:-1]


def abiertos(s):
    """Solo los dias abiertos, una fila por dia."""
    return s.dropna()

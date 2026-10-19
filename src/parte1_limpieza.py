"""Parte 1: limpieza.

Reporta el tamano de los datos, cuantifica cada caso sospechoso, arma la
bitacora de limpieza y revisa las seis lineas mas grandes antes de limpiar.

Uso:  python3 src/parte1_limpieza.py
"""

import pandas as pd

from comun import escribir_tex, macro, miles, libras, pct
from limpieza import cargar, limpiar, NO_PRODUCTO, REGLAS


def tex_texto(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main():
    crudo = cargar()
    crudo["Monto"] = crudo["Quantity"] * crudo["UnitPrice"]

    # 1.1
    filas = len(crudo)
    facturas = crudo["InvoiceNo"].nunique()
    productos = crudo["StockCode"].nunique()
    clientes = crudo["CustomerID"].nunique()
    sin_cliente = crudo["CustomerID"].isna().mean()
    print(f"filas={filas} facturas={facturas} productos={productos} "
          f"clientes={clientes} sin CustomerID={sin_cliente:.4f}")

    # 1.2: cada caso medido sobre los datos crudos, por separado
    es_c = crudo["InvoiceNo"].str.startswith("C")
    no_cinco = ~crudo["StockCode"].str.match(r"^\d{5}")
    casos = {
        "Duplicadas exactas": crudo.duplicated(),
        "Prefijo C (cancelación)": es_c,
        "Prefijo A (deuda incobrable)": crudo["InvoiceNo"].str.startswith("A"),
        "StockCode sin cinco dígitos": no_cinco,
        "\\quad de ellos, servicios eliminados": crudo["StockCode"].isin(NO_PRODUCTO),
        "\\quad de ellos, M y D (ajustes de precio)": crudo["StockCode"].isin(["M", "m", "D"]),
        "Precio cero": crudo["UnitPrice"] == 0,
        "Precio negativo": crudo["UnitPrice"] < 0,
        "Cantidad negativa sin C": (crudo["Quantity"] < 0) & ~es_c,
        "Sin CustomerID": crudo["CustomerID"].isna(),
    }
    lineas = []
    for nombre, m in casos.items():
        lineas.append(f"{nombre} & ${miles(m.sum())}$ & ${libras(crudo.loc[m, 'Monto'].sum())}$ \\\\")
        print(f"  {nombre:<45} {m.sum():>7} {crudo.loc[m, 'Monto'].sum():>14,.2f}")
    escribir_tex("tabla_p1_casos.tex", lineas)

    # Codigos que no empiezan con cinco digitos
    cod = crudo[no_cinco].copy()
    cod["Familia"] = cod["StockCode"].str.replace(r"^(DCGS|gift_0001).*", r"\1", regex=True)
    resumen = (cod.groupby("Familia")
               .agg(n=("Monto", "size"), monto=("Monto", "sum"),
                    desc=("Description", lambda s: s.dropna().iloc[0] if s.notna().any() else ""))
               .sort_values("n", ascending=False))
    lineas = []
    for fam, r in resumen.iterrows():
        if fam in NO_PRODUCTO:
            decision = "se va"
        elif fam in ("M", "m", "D"):
            decision = "ajuste, se queda"
        elif fam == "B":
            decision = "se va (prefijo A)"
        else:
            decision = "producto, se queda"
        etiqueta = {"DCGS": "DCGS\\ldots", "gift_0001": "gift\\_0001\\_\\ldots"}.get(fam, tex_texto(fam))
        lineas.append(f"\\code{{{etiqueta}}} & {tex_texto(r['desc'])} & ${miles(r['n'])}$ & "
                      f"${libras(r['monto'])}$ & {decision} \\\\")
    escribir_tex("tabla_p1_codigos.tex", lineas)

    # Descripciones de las cantidades negativas sin C
    neg = crudo[(crudo["Quantity"] < 0) & ~es_c]
    top_desc = neg["Description"].fillna("(vacía)").value_counts().head(8)
    print("  descripciones de negativas sin C:", dict(top_desc))

    # 1.3: bitacora
    bitacora = []
    limpio = limpiar(crudo.drop(columns="Monto"), bitacora=bitacora)
    lineas = []
    for i, b in enumerate(bitacora, 1):
        lineas.append(f"{i} & {b['regla']} & ${miles(b['filas'])}$ & ${libras(b['monto'], 2)}$ \\\\")
        print(f"  regla {i}: {b['filas']:>6} filas  {b['monto']:>14,.2f}")
    quitadas = sum(b["filas"] for b in bitacora)
    lineas.append("\\midrule")
    lineas.append(f" & Total eliminado & ${miles(quitadas)}$ & "
                  f"${libras(sum(b['monto'] for b in bitacora), 2)}$ \\\\")
    escribir_tex("tabla_p1_bitacora.tex", lineas)

    # 1.4: las seis lineas mas grandes antes de limpiar
    top = crudo.reindex(crudo["Monto"].abs().sort_values(ascending=False).index).head(6)
    claves = set(zip(limpio["InvoiceNo"], limpio["StockCode"], limpio["Quantity"]))
    lineas = []
    for idx, r in top.iterrows():
        queda = (r["InvoiceNo"], r["StockCode"], r["Quantity"]) in claves
        dia = r["InvoiceDate"].normalize()
        mismo = limpio[(limpio["CustomerID"] == r["CustomerID"]) &
                       (limpio["InvoiceDate"].dt.normalize() == dia)]
        lineas.append(f"\\code{{{r['InvoiceNo']}}} & {r['InvoiceDate']:%d/%m/%Y %H:%M} & "
                      f"{tex_texto(r['Description'])} & ${miles(r['Quantity'])}$ & "
                      f"${libras(r['UnitPrice'], 2)}$ & ${libras(r['Monto'], 2)}$ & "
                      f"{r['CustomerID']} & {'sí' if queda else 'no'} & "
                      f"${libras(mismo['Monto'].sum(), 2)}$ \\\\")
        print(f"  top: {r['InvoiceNo']} {r['Description']} {r['Monto']:,.2f} queda={queda} "
              f"neto cliente-dia={mismo['Monto'].sum():,.2f}")
    escribir_tex("tabla_p1_top6.tex", lineas)

    escribir_tex("datos_p1.tex", [
        macro("FilasCrudas", miles(filas)),
        macro("Facturas", miles(facturas)),
        macro("Productos", miles(productos)),
        macro("Clientes", miles(clientes)),
        macro("FracSinCliente", pct(sin_cliente)),
        macro("FilasLimpias", miles(len(limpio))),
        macro("FilasQuitadas", miles(quitadas)),
        macro("FracQuitadas", pct(quitadas / filas)),
        macro("FilasNegSinC", miles(len(neg))),
        macro("NegSinCConCliente", miles(neg["CustomerID"].notna().sum())),
        macro("NegSinCPrecioMax", f"{neg['UnitPrice'].max():g}"),
        macro("FilasCancel", miles(int(es_c.sum()))),
        macro("MontoCancel", libras(crudo.loc[es_c, "Monto"].sum())),
        macro("VentaNetaLimpia", libras(limpio["Monto"].sum())),
        macro("VentaNetaCruda", libras(crudo["Monto"].sum())),
    ])
    print(f"filas limpias={len(limpio)}  venta neta={limpio['Monto'].sum():,.2f}")


if __name__ == "__main__":
    main()

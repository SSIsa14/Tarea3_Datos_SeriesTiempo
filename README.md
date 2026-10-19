# Tarea 3: Introducción a Series de Tiempo

Análisis y Ciencia de Datos
Facultad de Ingeniería, Universidad del Istmo de Guatemala

Simulación de procesos AR(1) y de procesos con media cambiante (Parte 0), y
análisis de la venta neta diaria de Online Retail: limpieza, calendario,
descomposición, modelos AR(1) y SARIMA, pronóstico con origen móvil y
análisis RFM (Partes 1 a 6). La autocorrelación y Ljung–Box están hechos a
mano con numpy.

Todo el documento se reproduce corriendo siete scripts y compilando el `.tex`.
Abajo está el procedimiento completo.

---

## 1. Requisitos

- **Python 3.12**, que es la versión con la que se probó todo. Verificar con:

  ```bash
  python3 --version
  ```

- **Las librerías de Python.** Conviene instalarlas en un entorno virtual:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install numpy pandas matplotlib scipy statsmodels scikit-learn openpyxl
  ```

  Se probó con numpy 2.5, pandas 3.0, matplotlib 3.11, statsmodels 0.15 y
  scikit-learn 1.9. `openpyxl` sólo hace falta para leer el `.xlsx`.

- **Los datos.** El archivo `Online Retail.xlsx` del repositorio de UCI no
  viene en el repositorio. Hay que descargarlo y dejarlo en `data/`:

  ```bash
  mkdir -p data
  curl -L -o online_retail.zip "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
  unzip online_retail.zip -d data
  ```

  Debe quedar como `data/Online Retail.xlsx` (con el espacio en el nombre).

- **Una distribución de LaTeX**, para compilar el PDF. Sirve cualquiera de
  estas opciones:

  ```bash
  sudo apt install texlive-latex-extra texlive-lang-spanish   # opción 1
  ```

  ```bash
  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh   # opción 2: tectonic
  ```

  La opción 3 es subir la carpeta `doc/` junto con `figs/` a Overleaf y
  compilar ahí. El PDF de este repositorio se compiló con tectonic.

---

## 2. Reproducir todo, paso a paso

Los comandos se corren **desde la raíz del proyecto** (la carpeta donde está
este archivo), con el entorno virtual activado.

### Paso 1. Parte 0, procesos simulados

```bash
python3 src/parte0_simulacion.py
```

Tarda unos 3 segundos. Simula 5000 réplicas de cada escenario. En pantalla
debe aparecer, entre otras cosas:

- `phi=0.0  r1=-0.0022 ... rechazo=0.0462`
- `Ljung-Box propio Q=33.654177 p=0.000211 | statsmodels Q=33.654177 p=0.000211`

### Paso 2. Parte 1, limpieza

```bash
python3 src/parte1_limpieza.py
```

Tarda unos 25 segundos la primera vez, porque lee el `.xlsx`. Deja una copia
en `data/online_retail.pkl` para que los demás scripts no lo vuelvan a leer;
si el `.xlsx` cambia, la copia se regenera sola. Debe terminar con:

```
filas limpias=531877  venta neta=9,697,276.11
```

### Paso 3. Parte 2, el calendario

```bash
python3 src/parte2_calendario.py
```

Tarda unos 2 segundos. Debe reportar `dias calendario=374  con transacciones=305`
y listar los 18 días cerrados que no son sólo sábado.

### Paso 4. Parte 3, tendencia, estacionalidad y ACF

```bash
python3 src/parte3_estacionalidad.py
```

Tarda unos 2 segundos. El pico de la ACF debe salir en el rezago 6.

### Paso 5. Parte 4, del AR(1) al SARIMA

```bash
python3 src/parte4_sarima.py
```

Tarda unos 4 segundos. S1 debe salir `convergio=no` con 50 iteraciones y
`convergio=sí` con 500: es el resultado que se analiza en el documento.

### Paso 6. Parte 5, pronóstico y fuga

```bash
python3 src/parte5_pronostico.py
```

Tarda unos 20 segundos, que es el más lento porque reajusta los SARIMA en
cada uno de los 32 orígenes. S2 debe salir con el menor MAE en prueba
(`MAE 12,545`) y los SARIMA deben converger en `32 de 32` orígenes.

### Paso 7. Parte 6, RFM

```bash
python3 src/parte6_rfm.py
```

Tarda unos 3 segundos. Debe reportar `clientes=3615` y los AUC de R, F, M y
de la suma de puntajes.

### Paso 8. Compilar el documento

```bash
cd doc
tectonic -X compile tarea3.tex
```

Con una instalación normal de LaTeX, el equivalente es correr `pdflatex
tarea3.tex` **dos veces** (la primera pasada arma el índice y las
referencias, y la segunda las coloca bien).

El resultado es `doc/tarea3.pdf`, de 29 páginas.

---

## 3. Qué produce cada script

Los scripts escriben sus figuras en `figs/` y sus tablas en `doc/`. El `.tex`
las lee de ahí, así que hay que correrlos antes de compilar.

| Script | Figuras | Tablas |
|---|---|---|
| `parte0_simulacion.py` | `p0_realizaciones.png`, `p0_media_cambiante.png` | `tabla_p0_*.tex` (3), `datos_p0.tex` |
| `parte1_limpieza.py` | — | `tabla_p1_*.tex` (4, incluida la bitácora), `datos_p1.tex` |
| `parte2_calendario.py` | `p2_calendario.png` | `tabla_p2_cerrados.tex`, `datos_p2.tex` |
| `parte3_estacionalidad.py` | `p3_tendencia.png`, `p3_indices.png`, `p3_acf.png` | `tabla_p3_indices.tex`, `datos_p3.tex` |
| `parte4_sarima.py` | `p4_acf_residuos.png`, `p4_horizonte.png` | `tabla_p4_*.tex` (2), `datos_p4.tex` |
| `parte5_pronostico.py` | `p5_prueba.png` | `tabla_p5_errores.tex`, `datos_p5.tex` |
| `parte6_rfm.py` | `p6_tasas.png`, `p6_reparto.png` | `tabla_p6_*.tex` (2), `datos_p6.tex` |

Los archivos que empiezan con `datos_` sólo contienen definiciones de valores
y se leen desde el preámbulo del `.tex`. Ninguno de los archivos generados se
edita a mano: se sobrescriben en cada ejecución.

---

## 4. Semillas

Están fijas dentro de cada script, así que dos corridas distintas dan
exactamente los mismos resultados.

| Semilla | Dónde se usa |
|---|---|
| 140405 | Parte 0, AR(1) con φ = 0 |
| 77714 | Parte 0, AR(1) con φ = 0.5 |
| 141903 | Parte 0, AR(1) con φ = 0.9 |
| 1404057 | Parte 0, caminata aleatoria (φ = 1) |
| 777141 | Parte 0, proceso con tendencia lineal |
| 1419031 | Parte 0, proceso con patrón semanal |
| 141903 a 141912 | Parte 5, las 10 repeticiones de la validación cruzada aleatoria |
| 2026 | Parte 6, desempate de los puntajes RFM (la fija el enunciado) |

---

## 5. La función `limpiar`

Está en su propio archivo, `src/limpieza.py`, para poder reusarla en el
proyecto:

```python
from limpieza import cargar, limpiar

crudo = cargar()          # lee data/Online Retail.xlsx con los tipos correctos
limpio = limpiar(crudo)   # aplica las cinco reglas y agrega la columna Monto
```

Si se le pasa una lista en `bitacora=`, le agrega una fila por regla con las
filas y el monto que elimina cada una.

---

## 6. Si algo falla

**`FileNotFoundError: ... Online Retail.xlsx`**
Falta descargar los datos (sección 1). El archivo va en `data/` con su nombre
original.

**`ModuleNotFoundError: No module named 'pandas'`** (u otra librería)
Falta instalar las dependencias o activar el entorno virtual.

**`ModuleNotFoundError: No module named 'comun'`**
Se está corriendo el script desde otra carpeta. Los comandos van desde la
raíz del proyecto, en la forma `python3 src/nombre.py`.

**El PDF compila pero faltan tablas o aparece `Undefined control sequence`**
Faltó correr algún script antes de compilar. Hay que hacer los pasos 1 al 7 y
volver a compilar.

---

## 7. Estructura del proyecto

```
src/
  comun.py                  rutas, salida a LaTeX y estilo de las gráficas
  series.py                 ACF, Ljung-Box y OLS del AR(1), hechos a mano
  limpieza.py               carga del .xlsx y la función limpiar
  datos.py                  serie diaria y partición entrenamiento/prueba
  parte0_simulacion.py      Parte 0
  parte1_limpieza.py        Parte 1
  parte2_calendario.py      Parte 2
  parte3_estacionalidad.py  Parte 3
  parte4_sarima.py          Parte 4
  parte5_pronostico.py      Parte 5
  parte6_rfm.py             Parte 6
doc/
  tarea3.tex                el documento
  tarea3.pdf                el PDF compilado
  (tablas generadas por los scripts)
figs/                       figuras generadas por los scripts
data/                       Online Retail.xlsx (no se sube al repositorio)
README.md
```

Correr los siete scripts toma alrededor de un minuto en total.

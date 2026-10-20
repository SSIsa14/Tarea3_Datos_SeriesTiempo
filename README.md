# Tarea 3: Introducción a Series de Tiempo

Análisis y Ciencia de Datos
Facultad de Ingeniería, Universidad del Istmo de Guatemala

Simulación de procesos AR(1) y de procesos con media cambiante (Parte 0), y
análisis de la venta neta diaria de Online Retail: limpieza, calendario,
descomposición, modelos AR(1) y SARIMA, pronóstico con origen móvil y
análisis RFM (Partes 1 a 6).

El documento final es [`doc/tarea3.pdf`](doc/tarea3.pdf) y el enunciado es
[`BI_Tarea4.pdf`](BI_Tarea4.pdf). Todo lo que aparece en el documento (cifras,
tablas y figuras) se genera con los scripts de `src/` a partir del archivo
`.xlsx` original.

---

## Resumen rápido

Para quien ya tiene Python y quiere correr todo de una vez (Linux o macOS):

```bash
git clone https://github.com/SSIsa14/Tarea3_Datos_SeriesTiempo.git
cd Tarea3_Datos_SeriesTiempo
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy statsmodels scikit-learn openpyxl
mkdir -p data
curl -L -o data/online_retail.zip "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
unzip data/online_retail.zip -d data
for s in parte0_simulacion parte1_limpieza parte2_calendario parte3_estacionalidad parte4_sarima parte5_pronostico parte6_rfm; do python3 src/$s.py; done
cd doc && tectonic -X compile tarea3.tex
```

Abajo está cada paso explicado.

---

## Paso a paso

### Paso 1. Descargar el repositorio

```bash
git clone https://github.com/SSIsa14/Tarea3_Datos_SeriesTiempo.git
cd Tarea3_Datos_SeriesTiempo
```

**Todos los comandos que siguen se corren desde esta carpeta** (la raíz del
proyecto, donde está este README). Si un script se corre desde otra carpeta,
no encuentra sus módulos.

### Paso 2. Revisar que haya Python

```bash
python3 --version
```

Debe decir `Python 3.10` o mayor. Todo se probó con Python 3.12. En Windows
el comando suele ser `python` en lugar de `python3`.

### Paso 3. Crear el entorno virtual e instalar las librerías

El entorno virtual es una carpeta (`.venv`) donde se instalan las librerías
sólo para este proyecto, sin tocar el resto de la computadora.

```bash
python3 -m venv .venv
```

Después hay que **activarlo**:

```bash
source .venv/bin/activate        # Linux o macOS
.venv\Scripts\activate           # Windows
```

Cuando está activo, la terminal muestra `(.venv)` al inicio de la línea. Hay
que activarlo cada vez que se abre una terminal nueva para trabajar en el
proyecto.

Con el entorno activo, se instalan las librerías:

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn openpyxl
```

Se probó con numpy 2.5, pandas 3.0, matplotlib 3.11, statsmodels 0.15 y
scikit-learn 1.9. `openpyxl` es la que permite leer el archivo de Excel.

### Paso 4. Descargar los datos

Los datos no vienen en el repositorio porque son públicos. Son el archivo
`Online Retail.xlsx` del repositorio de UCI, y tienen que quedar dentro de
una carpeta `data/`.

**Opción A, con la terminal (Linux o macOS):**

```bash
mkdir -p data
curl -L -o data/online_retail.zip "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
unzip data/online_retail.zip -d data
```

**Opción B, a mano (cualquier sistema):**

1. Entrar a <https://archive.ics.uci.edu/dataset/352/online+retail> y darle a
   *Download*.
2. Descomprimir el `.zip`.
3. Crear la carpeta `data` en la raíz del proyecto y mover ahí el archivo.

En cualquiera de las dos opciones tiene que quedar así, con el nombre
original (incluido el espacio):

```
data/Online Retail.xlsx
```

### Paso 5. Correr los scripts

Hay un script por cada parte de la tarea. Se corren **en este orden**, con el
entorno virtual activo:

| # | Comando | Qué hace | Tiempo aprox. |
|---|---|---|---|
| 1 | `python3 src/parte0_simulacion.py` | Parte 0: simula los AR(1) y los procesos con media cambiante | 3 s |
| 2 | `python3 src/parte1_limpieza.py` | Parte 1: limpieza de los datos y bitácora | 25 s |
| 3 | `python3 src/parte2_calendario.py` | Parte 2: serie diaria y días cerrados | 2 s |
| 4 | `python3 src/parte3_estacionalidad.py` | Parte 3: tendencia, índices estacionales y ACF | 2 s |
| 5 | `python3 src/parte4_sarima.py` | Parte 4: indicadoras, AR(1) y los SARIMA S1 y S2 | 4 s |
| 6 | `python3 src/parte5_pronostico.py` | Parte 5: los seis modelos y las tres validaciones | 20 s |
| 7 | `python3 src/parte6_rfm.py` | Parte 6: análisis RFM | 3 s |

El script 2 es el único que lee el Excel, y por eso es el más lento la
primera vez. Guarda una copia en `data/online_retail.pkl` para que los
demás scripts arranquen rápido.

Para correr los siete seguidos con un solo comando (Linux o macOS):

```bash
for s in parte0_simulacion parte1_limpieza parte2_calendario parte3_estacionalidad parte4_sarima parte5_pronostico parte6_rfm; do python3 src/$s.py; done
```

Cada script imprime en la terminal sus resultados principales, para poder
compararlos con el documento. Algunos valores que deben aparecer:

| Script | Debe aparecer |
|---|---|
| `parte0_simulacion.py` | `phi=0.0  r1=-0.0022 ... rechazo=0.0462` |
| `parte1_limpieza.py` | `filas limpias=531877  venta neta=9,697,276.11` |
| `parte2_calendario.py` | `dias calendario=374  con transacciones=305` |
| `parte3_estacionalidad.py` | `... pico=6 ...` |
| `parte4_sarima.py` | S1 con `convergio=no` en 50 iteraciones y `convergio=sí` en 500 |
| `parte5_pronostico.py` | `S2: ... (c) MAE 12,545 ...` y `convergio en 32 de 32 origenes` |
| `parte6_rfm.py` | `clientes=3615` |

Al terminar, las figuras quedan en `figs/` y las tablas en `doc/`.

### Paso 6. Compilar el documento

El documento en LaTeX lee las figuras y tablas que generaron los scripts, así
que este paso va después del paso 5.

**Opción A, con tectonic** (así se compiló el PDF del repositorio):

```bash
cd doc
tectonic -X compile tarea3.tex
```

Si no está instalado:

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
```

Ese comando deja el programa `tectonic` en la carpeta donde se corre.

**Opción B, con una instalación normal de LaTeX** (TeX Live o MiKTeX):

```bash
cd doc
pdflatex tarea3.tex
pdflatex tarea3.tex
```

Se corre **dos veces**: la primera arma el índice y las referencias, y la
segunda las coloca bien.

**Opción C, en Overleaf:** subir la carpeta `doc/` y la carpeta `figs/` y
compilar ahí.

El resultado es `doc/tarea3.pdf`, de 29 páginas.

---

## Semillas

Están fijas dentro de cada script, así que dos corridas dan exactamente los
mismos resultados.

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

## Qué produce cada script

| Script | Figuras (`figs/`) | Tablas (`doc/`) |
|---|---|---|
| `parte0_simulacion.py` | `p0_realizaciones.png`, `p0_media_cambiante.png` | `tabla_p0_*.tex` (3), `datos_p0.tex` |
| `parte1_limpieza.py` | — | `tabla_p1_*.tex` (4, incluida la bitácora), `datos_p1.tex` |
| `parte2_calendario.py` | `p2_calendario.png` | `tabla_p2_cerrados.tex`, `datos_p2.tex` |
| `parte3_estacionalidad.py` | `p3_tendencia.png`, `p3_indices.png`, `p3_acf.png` | `tabla_p3_indices.tex`, `datos_p3.tex` |
| `parte4_sarima.py` | `p4_acf_residuos.png`, `p4_horizonte.png` | `tabla_p4_*.tex` (2), `datos_p4.tex` |
| `parte5_pronostico.py` | `p5_prueba.png` | `tabla_p5_errores.tex`, `datos_p5.tex` |
| `parte6_rfm.py` | `p6_tasas.png`, `p6_reparto.png` | `tabla_p6_*.tex` (2), `datos_p6.tex` |

Los archivos `tabla_*.tex` tienen las filas de cada tabla y los `datos_*.tex`
tienen los valores que se citan en el texto. Cada vez que se corre un script,
sus archivos se vuelven a generar.

---

## La función `limpiar`

Está en su propio archivo, `src/limpieza.py`, para poder reusarla en el
proyecto. Desde un script dentro de `src/`:

```python
from limpieza import cargar, limpiar

crudo = cargar()          # lee data/Online Retail.xlsx con los tipos correctos
limpio = limpiar(crudo)   # aplica las cinco reglas y agrega la columna Monto
```

Si se le pasa una lista en `bitacora=`, le agrega una fila por regla con las
filas y el monto que elimina cada una.

---

## Si algo falla

**`FileNotFoundError: ... Online Retail.xlsx`**
Faltan los datos o no están en el lugar correcto. Revisar el paso 4: el
archivo tiene que estar en `data/Online Retail.xlsx`.

**`ModuleNotFoundError: No module named 'pandas'`** (u otra librería)
El entorno virtual no está activo o faltó instalar las librerías. Revisar el
paso 3.

**`ModuleNotFoundError: No module named 'comun'`** (u otro módulo de `src/`)
El script se está corriendo desde otra carpeta. Hay que volver a la raíz del
proyecto y correrlo como `python3 src/nombre.py`.

**El PDF compila pero faltan tablas o aparece `Undefined control sequence`**
Faltó correr algún script antes de compilar. Hay que hacer el paso 5 completo
y volver a compilar.

---

## Estructura del proyecto

```
BI_Tarea4.pdf               enunciado de la tarea
README.md
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
  tabla_*.tex, datos_*.tex  tablas y valores generados por los scripts
figs/                       figuras generadas por los scripts
data/                       Online Retail.xlsx (se descarga en el paso 4)
```

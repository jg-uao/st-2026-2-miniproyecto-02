"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras que produce este guion.

Serie del grupo: demanda real mensual de energía eléctrica del Sistema
Interconectado Nacional de Colombia (GWh), publicada por XM, 2015-2025. Es
la misma del miniproyecto 1, y el modelo a superar es su SARIMAX con días
del mes y festivos.
"""
import datetime as dt
import time
import warnings
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# Los mismos datos del miniproyecto 1: la demanda mensual de energía de XM.
ARCHIVO = "datos/demanda_mensual.csv"
FECHA = "ts"
VALOR = "GWh"
PERIODO = 12
UNIDAD = "Demanda de energía (GWh)"

# Meses atípicos que se señalan en la figura y se conservan, como en el
# miniproyecto 1: la cuarentena de 2020 y el paro de mayo de 2021.
ATIPICOS = pd.to_datetime(["2020-04-01", "2020-05-01", "2020-06-01",
                           "2021-05-01"])


def preparar(serie):
    """La misma preparación del miniproyecto 1.

    La serie no tiene meses vacíos; por si el CSV cambiara, los huecos se
    rellenan con interpolación lineal. Los atípicos de abril-junio de 2020
    (cuarentena) y mayo de 2021 (paro) se conservan: son demanda real y
    quedan fuera de la ventana de prueba.
    """
    vacios = serie.isna().sum()
    if vacios:
        print(f"  {vacios} meses vacíos rellenados por interpolación lineal")
    return serie.interpolate(method="linear")


# ---------- El calendario: días del mes y festivos de Colombia ----------
# En el miniproyecto 1 los festivos salían del paquete holidays, que aquí no
# entra (solo se usan las siete bibliotecas del enunciado). Son las mismas
# fechas, calculadas con la regla: seis festivos de fecha fija, siete que la
# ley Emiliani pasa al lunes siguiente y cinco que dependen de la Pascua. De
# 2015 a 2025 coinciden día por día con las de holidays.

def pascua(anio):
    """Domingo de Pascua en el calendario gregoriano (algoritmo de Butcher)."""
    g, c = anio % 19, anio // 100
    h = (c - c // 4 - (8 * c + 13) // 25 + 19 * g + 15) % 30
    i = h - (h // 28) * (1 - (h // 28) * (29 // (h + 1)) * ((21 - g) // 11))
    j = (anio + anio // 4 + i + 2 - c + c // 4) % 7
    p = i - j
    mes = 3 + (p + 40) // 44
    return dt.date(anio, mes, p + 28 - 31 * (mes // 4))


@lru_cache(maxsize=None)
def festivos_del_anio(anio):
    def al_lunes(fecha):
        return fecha + dt.timedelta(days=(7 - fecha.weekday()) % 7)

    fijos = {dt.date(anio, m, d)
             for m, d in [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]}
    emiliani = {al_lunes(dt.date(anio, m, d))
                for m, d in [(1, 6), (3, 19), (6, 29), (8, 15), (10, 12), (11, 1), (11, 11)]}
    # Jueves y Viernes Santo, Ascensión, Corpus Christi y Sagrado Corazón.
    de_pascua = {pascua(anio) + dt.timedelta(days=k) for k in (-3, -2, 43, 64, 71)}
    return frozenset(fijos | emiliani | de_pascua)


def calendario(indice):
    """Días del mes y festivos de cada mes de `indice`. Las dos columnas se
    conocen años antes y ninguna depende de la demanda: no hay fuga."""
    festivos = [sum(f.month == mes.month for f in festivos_del_anio(mes.year))
                for mes in indice]
    return pd.DataFrame({"dias": indice.days_in_month.astype(float),
                         "festivos": np.asarray(festivos, dtype=float)}, index=indice)


# ---------- El mejor modelo del miniproyecto 1 ----------
# SARIMAX(1,1,1)(1,0,1,12) con días del mes y festivos, ajustado sobre la
# serie dividida entre ESCALA para que sigma2 quede en el orden de magnitud
# de los demás coeficientes. Los pronósticos salen en GWh.
ORDEN = (1, 1, 1)
ORDEN_ESTACIONAL = (1, 0, 1, PERIODO)
ESCALA = 10


def ajustar_sarimax(entrenamiento, exog):
    """El mismo ajuste del miniproyecto 1: si el optimizador no converge, se
    reintenta desde los parámetros alcanzados."""
    modelo = ARIMA(entrenamiento / ESCALA, exog=exog, order=ORDEN,
                   seasonal_order=ORDEN_ESTACIONAL)
    ajuste = modelo.fit()
    intentos = 0
    while not ajuste.mle_retvals.get("converged", True) and intentos < 2:
        ajuste = modelo.fit(start_params=ajuste.params, method_kwargs={"maxiter": 200})
        intentos += 1
    return ajuste


def modelo_clasico(serie):
    """La función modelo(entrenamiento, h) del SARIMAX del miniproyecto 1.

    Ajusta con las exógenas de los meses de entrenamiento y pronostica con
    las de los h meses siguientes, que son calendario.
    """
    x = calendario(pd.date_range(serie.index[0], periods=len(serie) + 12, freq="MS"))

    def modelo(entrenamiento, h):
        futuro = fechas_siguientes(entrenamiento, h)
        ajuste = ajustar_sarimax(entrenamiento, x.loc[entrenamiento.index])
        ULTIMO["modelo"] = ajuste
        return ajuste.forecast(h, exog=x.loc[futuro]).to_numpy() * ESCALA
    return modelo


NOMBRE_CLASICO = "SARIMAX(1,1,1)(1,0,1,12) + días y festivos"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy(), lw=1.2)
    eje.scatter(ATIPICOS, serie.loc[ATIPICOS], color="C3", zorder=3, s=25,
                label="atípicos conservados")
    eje.axvspan(serie.index[-N_PRUEBA], serie.index[-1], color="0.85", alpha=0.6,
                label="los 24 meses del backtest")
    eje.set_title(f"{UNIDAD}: serie mensual, SIN Colombia (XM)")
    eje.set_ylabel(VALOR)
    eje.legend(loc="upper left", fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior. Se elige "dif12": la serie
# crece 28 % en once años y un árbol o un KNN no pronostican por encima de
# lo que vieron. comparar_objetivos lo mide con los meses anteriores al
# backtest.
OBJETIVO = "dif12"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes t con lo que se sabe antes de observar z(t).

    - rezago_k = z(t-k), k = 1..12.
    - media_3, media_12 y desv_12: media y desviación móviles sobre la serie
      ya corrida un mes, así que terminan en t-1.
    - mes_sen y mes_cos: el mes del año, en el círculo.
    - dias y festivos: el calendario del mes t, que se conoce de antemano.
      Si el objetivo es la diferencia de 12, van también como diferencia con
      el mismo mes del año anterior (el día de más del bisiesto y la Semana
      Santa que cambia de mes).
    """
    X = pd.DataFrame(index=z.index)
    for k in range(1, REZAGOS + 1):
        X[f"rezago_{k}"] = z.shift(k)
    pasado = z.shift(1)
    X["media_3"] = pasado.rolling(3).mean()
    X["media_12"] = pasado.rolling(12).mean()
    X["desv_12"] = pasado.rolling(12).std()
    angulo = 2 * np.pi * z.index.month / PERIODO
    X["mes_sen"] = np.sin(angulo)
    X["mes_cos"] = np.cos(angulo)
    actual = calendario(z.index)
    if OBJETIVO == "dif12":
        actual = actual - calendario(z.index - pd.DateOffset(years=1)).to_numpy()
    X["dias"] = actual["dias"]
    X["festivos"] = actual["festivos"]
    return X


def tabla_con_fuga(z):
    """La misma tabla con un error a propósito: la media móvil sin correr la
    serie, que incluye z(t). Solo sirve para comprobar que la prueba de fuga
    detecta una fuga cuando la hay."""
    X = tabla_supervisada(z)
    X["media_3"] = z.rolling(3).mean()
    return X


def filas_que_ven_su_mes(serie, tabla):
    """Cambia el valor de cada mes y cuenta en cuántos cambia su propia fila."""
    original = tabla(objetivo(serie))
    cambian = 0
    for i in range(len(serie)):
        alterada = serie.copy()
        alterada.iloc[i] += 1000.0
        fila_nueva = tabla(objetivo(alterada)).iloc[i]
        if not fila_nueva.equals(original.iloc[i]):
            cambian += 1
    return cambian


def comprobar_sin_fuga(serie):
    """Prueba en el código que ninguna fila ve el mes que predice.

    A cada mes se le suman 1 000 GWh, se rehace la tabla y su fila tiene que
    quedar igual. Para saber que la prueba sirve, se repite con una tabla que
    sí tiene fuga y con la fila del mes siguiente, que sí debe cambiar.
    """
    cambian = filas_que_ven_su_mes(serie, tabla_supervisada)
    assert cambian == 0, f"{cambian} filas cambian al alterar su propio mes: hay fuga"
    con_fuga = filas_que_ven_su_mes(serie, tabla_con_fuga)
    assert con_fuga > 0, "la prueba no detecta una tabla con fuga"
    # La fila t+1 sí depende de z(t): si no cambiara, la prueba no mediría nada.
    alterada = serie.copy()
    alterada.iloc[-2] += 1000.0
    siguiente = tabla_supervisada(objetivo(alterada)).iloc[-1]
    assert not siguiente.equals(tabla_supervisada(objetivo(serie)).iloc[-1])
    X, _ = filas_completas(serie)
    print(f"  tabla: {X.shape[0]} filas completas y {X.shape[1]} columnas, "
          f"de {X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}; objetivo: {OBJETIVO}")
    previos = serie.iloc[:-N_PRUEBA]
    z = objetivo(serie)
    print(f"  {int((serie.iloc[-N_PRUEBA:] > previos.max()).sum())} de los {N_PRUEBA} meses de "
          f"prueba superan el máximo anterior ({previos.max():.0f} GWh en {previos.idxmax():%Y-%m})")
    if OBJETIVO == "dif12":
        print(f"  diferencia de 12: media {z.mean():.1f} y desviación {z.std():.1f} GWh; media "
              + ", ".join(f"{a}: {z[a].mean():+.1f}" for a in ("2022", "2023", "2024", "2025")))
    print(f"  sin fuga: al alterar cada uno de los {len(serie)} meses, {cambian} filas "
          f"propias cambian; con la tabla con fuga a propósito cambian {con_fuga}")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    En cada paso se agrega el mes siguiente como NaN, se arma su fila con lo
    que hay hasta el mes anterior, se pronostica el objetivo, se devuelve al
    nivel y ese pronóstico queda escrito como si fuera dato: es el rezago 1
    del mes siguiente.
    """
    y = y.copy()
    pronosticos = []
    for _ in range(h):
        conocido = y
        y = pd.concat([y, pd.Series([np.nan], index=fechas_siguientes(y, 1))])
        ultima = tabla_supervisada(objetivo(y)).iloc[[-1]]
        z_hat = float(np.asarray(ajustado.predict(ultima))[0])
        y.iloc[-1] = a_nivel(conocido, z_hat)
        pronosticos.append(y.iloc[-1])
    return np.asarray(pronosticos)


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


def comparar_objetivos(serie):
    """Nivel contra diferencia de 12, medido sin tocar los meses del backtest.

    Corre el backtest común sobre la serie recortada (sin los últimos 24
    meses), con los cuatro modelos de aprendizaje automático en sus valores
    por defecto y con cada objetivo. Guarda resultados/objetivo.csv.
    """
    global OBJETIVO
    elegido = OBJETIVO
    previa = serie.iloc[:-N_PRUEBA]
    filas = []
    for opcion in ("nivel", "dif12"):
        OBJETIVO = opcion
        for nombre, (fabrica, _) in REJILLAS.items():
            _, promedio = backtest(previa, con_tabla(fabrica))
            filas.append({"modelo": nombre, "objetivo": opcion,
                          "RMSE": round(promedio["RMSE"], 2)})
    OBJETIVO = elegido
    ULTIMO.clear()
    tabla = pd.DataFrame(filas).pivot(index="modelo", columns="objetivo", values="RMSE")
    tabla = tabla.loc[list(REJILLAS), ["nivel", "dif12"]]
    tabla.to_csv(RESULTADOS / "objetivo.csv")
    print(f"  RMSE con cada objetivo, backtest sobre {previa.index[0]:%Y-%m} a "
          f"{previa.index[-1]:%Y-%m} (sin los {N_PRUEBA} meses de prueba):")
    print("  " + tabla.to_string().replace("\n", "\n  "))


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# De dos a cuatro valores por hiperparámetro. El objetivo de la SVR no se
# escala, así que C y epsilon están en GWh: epsilon es el ancho del tubo sin
# penalización y C tiene que ser del orden de las diferencias (cientos).
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [3, 5, 9, 15]}),
    "SVR": (svr, {"C": [10.0, 100.0, 1000.0], "epsilon": [1.0, 10.0, 50.0]}),
    "Random Forest": (bosque, {"max_depth": [3, 6, None], "min_samples_leaf": [1, 3, 5]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1, 0.3]}),
}
PLIEGUES = 4           # TimeSeriesSplit: 4 cortes con 12 meses de validación cada uno


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """Para cada modelo de REJILLAS, la combinación con menor RMSE en
    validación temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA]: ninguno de los 24 meses del
    backtest decide. TimeSeriesSplit valida siempre con meses posteriores a
    los de ajuste, y el escalador de KNN y SVR, que vive en el pipeline, se
    ajusta dentro de cada pliegue.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    cortes = TimeSeriesSplit(n_splits=PLIEGUES, test_size=PERIODO)
    print(f"  rejilla con {len(X)} filas, de {X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}; "
          f"{PLIEGUES} pliegues de {PERIODO} meses")
    elegidos, filas = {}, []
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        modelo = fabrica()
        prefijo = f"{modelo.steps[-1][0]}__" if hasattr(modelo, "steps") else ""
        busqueda = GridSearchCV(modelo, {prefijo + k: v for k, v in rejilla.items()},
                                cv=cortes, scoring="neg_root_mean_squared_error")
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        puntajes = -busqueda.cv_results_["mean_test_score"]
        filas.append({"modelo": nombre, "elegido": str(elegidos[nombre]),
                      "RMSE_validacion": round(-busqueda.best_score_, 2),
                      "peor_de_la_rejilla": round(puntajes.max(), 2),
                      "combinaciones": len(puntajes)})
        print(f"  {nombre}: {elegidos[nombre]}, RMSE de validación "
              f"{-busqueda.best_score_:.2f} (la peor combinación de {len(puntajes)}: "
              f"{puntajes.max():.2f})")
    pd.DataFrame(filas).to_csv(RESULTADOS / "hiperparametros.csv", index=False)
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost
    (feature_importances_), ajustados con lo elegido sobre toda la serie."""
    X, z = filas_completas(serie)
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    todas = {}
    for eje, (nombre, fabrica) in zip(ejes, [("Random Forest", bosque), ("XGBoost", xgboost)]):
        ajustado = fabrica(**elegidos[nombre]).fit(X, z)
        importancia = pd.Series(ajustado.feature_importances_, index=X.columns)
        todas[nombre] = importancia
        eje.barh(importancia.index[::-1], importancia.to_numpy()[::-1])
        eje.set_title(f"{nombre}: importancia de las variables")
        mayores = importancia.sort_values(ascending=False).head(5)
        print(f"  {nombre}, las cinco de más peso: "
              + ", ".join(f"{k} {v:.3f}" for k, v in mayores.items())
              + f"; dias {importancia['dias']:.3f}, festivos {importancia['festivos']:.3f}")
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    pd.DataFrame(todas).round(3).rename_axis("variable").to_csv(RESULTADOS / "importancia.csv")
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


# Las 6 columnas que lee el MLP corto: los dos rezagos que usa el SARIMAX
# (el 1 y el 12, los de la PACF del miniproyecto 1), el mes y el calendario.
ENTRADAS_CORTAS = ["rezago_1", "rezago_12", "mes_sen", "mes_cos", "dias", "festivos"]


class MLPCorto(nn.Module):
    """El mismo MLP, que de las columnas de la tabla solo lee ENTRADAS_CORTAS.

    La tabla no cambia: la red recibe sus 19 columnas ya escaladas y se
    queda con 6. Con 72 a 96 filas, los doce rezagos y las ventanas, muy
    parecidos entre sí, le dan a la red con qué aprenderse el entrenamiento
    de memoria.
    """

    def __init__(self, entradas, ocultas=16):
        super().__init__()
        meses = pd.date_range("2000-01-01", periods=2, freq="MS")
        columnas = list(tabla_supervisada(pd.Series(0.0, index=meses)).columns)
        assert len(columnas) == entradas
        self.indices = [columnas.index(nombre) for nombre in ENTRADAS_CORTAS]
        self.mlp = MLP(len(self.indices), ocultas)

    def forward(self, x):
        return self.mlp(x[:, self.indices])


class LSTM(nn.Module):
    """nn.LSTM de una capa sobre la secuencia (lote, 12, 1) y una capa lineal
    que lee el último estado oculto."""

    def __init__(self, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=ocultas, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        estados, _ = self.lstm(x)
        return self.salida(estados[:, -1, :])


class Conv1D(nn.Module):
    """La tercera red: una convolución de una dimensión sobre los 12 rezagos.

    Cada filtro mira tres meses seguidos y se desliza por la secuencia: los
    mismos pesos sirven para cualquier trimestre, así que la red tiene pocos
    parámetros para los pocos meses que hay. Una capa lineal lee las diez
    posiciones de salida de cada filtro.
    """

    def __init__(self, filtros=8, nucleo=3):
        super().__init__()
        self.conv = nn.Conv1d(1, filtros, kernel_size=nucleo)
        self.salida = nn.Linear(filtros * (REZAGOS - nucleo + 1), 1)

    def forward(self, x):
        mapas = torch.relu(self.conv(x.transpose(1, 2)))   # (lote, 1, 12) -> (lote, filtros, 10)
        return self.salida(mapas.flatten(1))


# Las candidatas: nombre, cómo se construye y si recibe la secuencia. Hay dos
# familias de MLP, el que lee las 19 columnas y el corto, que lee 6; la
# validación deja un tamaño de cada una. La LSTM y la Conv1D reciben los 12
# rezagos como secuencia.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (16 ocultas)": (lambda n: MLP(n, 16), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "MLP corto (8 ocultas)": (lambda n: MLPCorto(n, 8), False),
    "MLP corto (32 ocultas)": (lambda n: MLPCorto(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(16), True),
    "Conv1D (8 filtros)": (lambda n: Conv1D(8), True),
}


def validar_redes(serie):
    """Las épocas de cada red y el tamaño del MLP, con validación temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA] y sus últimos 12 meses son la
    validación. Cada candidata se entrena EPOCAS_MAXIMAS épocas y se queda
    con la época de menor pérdida de validación. De cada familia de MLP queda
    el tamaño de menor pérdida de validación. Las curvas van a
    figuras/perdida.png, y resultados/validacion_redes.csv guarda además con
    qué se compara esa pérdida.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent, X_val, z_val = X.iloc[:-PERIODO], z.iloc[:-PERIODO], X.iloc[-PERIODO:], z.iloc[-PERIODO:]
    print(f"  entrenamiento: {len(X_ent)} filas hasta {X_ent.index[-1]:%Y-%m}; "
          f"validación: {X_val.index[0]:%Y-%m} a {X_val.index[-1]:%Y-%m}")
    # Con qué se compara la pérdida, en la misma escala: pronosticar la media
    # del entrenamiento, repetir la diferencia del mes anterior y crecimiento
    # cero, que es el estacional ingenuo.
    escalado = lambda p: float(np.mean(((p - z_val) / z_ent.std()) ** 2))  # noqa: E731
    referencias = {"media del entrenamiento": escalado(z_ent.mean()),
                   "repetir el mes anterior": escalado(X_val["rezago_1"]),
                   "crecimiento cero": escalado(0.0)}
    print("  referencias de la pérdida de validación: "
          + ", ".join(f"{k} {v:.3f}" for k, v in referencias.items()))
    resumen = {}
    columnas = 4
    figura, cuadros = plt.subplots(2, columnas, figsize=(3.3 * columnas, 6.2), sharey=True)
    ejes = cuadros.ravel()
    for eje in ejes[len(CANDIDATAS):]:
        eje.axis("off")
    for eje, (nombre, (construir, secuencia)) in zip(ejes, CANDIDATAS.items()):
        ajustada = Red(construir, secuencia).fit(X_ent, z_ent, X_val, z_val)
        curvas = np.asarray(ajustada.perdidas)
        mejor = int(curvas[:, 1].argmin())
        parametros = sum(p.numel() for p in ajustada.red.parameters())
        resumen[nombre] = {"epocas": mejor + 1, "validacion": curvas[mejor, 1],
                           "entrenamiento": curvas[mejor, 0], "parametros": parametros,
                           "validacion_final": curvas[-1, 1], "entrenamiento_final": curvas[-1, 0]}
        print(f"  {nombre}: {parametros} parámetros; mínimo de validación {curvas[mejor, 1]:.3f} "
              f"en la época {mejor + 1} (entrenamiento {curvas[mejor, 0]:.3f}); en la época "
              f"{EPOCAS_MAXIMAS}: validación {curvas[-1, 1]:.3f}, entrenamiento {curvas[-1, 0]:.3f}")
        eje.plot(range(1, len(curvas) + 1), curvas[:, 0], label="entrenamiento")
        eje.plot(range(1, len(curvas) + 1), curvas[:, 1], label="validación")
        eje.axvline(mejor + 1, color="gray", linestyle="--", lw=0.8)
        eje.axhline(referencias["media del entrenamiento"], color="gray", linestyle=":", lw=0.8)
        eje.set_yscale("log")
        eje.set_title(f"{nombre}\n{mejor + 1} épocas", fontsize=9)
        eje.set_xlabel("época")
    for fila_de_ejes in cuadros:
        fila_de_ejes[0].set_ylabel("MSE (objetivo escalado)")
    ejes[0].legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)
    tabla = pd.DataFrame(resumen).T.round(4)
    tabla.index.name = "red"
    tabla.to_csv(RESULTADOS / "validacion_redes.csv")
    pd.Series(referencias, name="validacion").round(4).rename_axis("referencia").to_csv(
        RESULTADOS / "validacion_referencias.csv")

    # De cada familia (el texto antes del paréntesis) queda la candidata de
    # menor pérdida de validación.
    familias = {}
    for nombre in resumen:
        familias.setdefault(nombre.split(" (")[0], []).append(nombre)
    elegidas = [min(nombres, key=lambda n: resumen[n]["validacion"]) for nombres in familias.values()]
    print("  elegidas: " + ", ".join(elegidas))
    return {nombre: int(resumen[nombre]["epocas"]) for nombre in elegidas}


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest.

    Guarda resultados/semillas.csv con una fila por semilla e imprime la
    media y la desviación del RMSE.
    """
    filas = []
    for semilla in range(SEMILLA, SEMILLA + 5):
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas.append({"red": nombre, "semilla": semilla, "RMSE": round(promedio["RMSE"], 2),
                      "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    print(f"  {nombre}, {epocas} épocas, cinco semillas:")
    print("  " + tabla.drop(columns="red").to_string(index=False).replace("\n", "\n  "))
    print(f"  RMSE: media {tabla['RMSE'].mean():.2f}, desviación {tabla['RMSE'].std():.2f}, "
          f"de {tabla['RMSE'].min():.2f} a {tabla['RMSE'].max():.2f}")
    return tabla


# ============================================================
# 5. La tabla y la elección
# ============================================================
# Lo que deja cada modelo al pasar por el backtest: su RMSE en cada origen y
# sus 12 pronósticos de cada origen, para ver dónde falla.
POR_ORIGEN = {}
PRONOSTICOS = {}


def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    guardados = []

    def registrado(entrenamiento, h):
        pronostico = np.asarray(modelo(entrenamiento, h))
        guardados.append(pd.Series(pronostico, index=fechas_siguientes(entrenamiento, h)))
        return pronostico

    inicio = time.perf_counter()
    por_origen, promedio = backtest(serie, registrado)
    segundos = time.perf_counter() - inicio
    POR_ORIGEN[nombre] = por_origen.set_index("origen")["RMSE"]
    PRONOSTICOS[nombre] = guardados
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def analizar_errores(serie, destacados):
    """Dónde falla cada modelo: por origen, por horizonte y por mes del año.

    El error de cada pronóstico es pronóstico menos real, así que un sesgo
    positivo es pronosticar de más. Guarda tres tablas en resultados/ y la
    figura figuras/errores.png con los modelos de `destacados`.
    """
    por_origen = pd.DataFrame(POR_ORIGEN).round(1)
    por_origen.index = por_origen.index.strftime("%Y-%m")
    por_origen.to_csv(RESULTADOS / "rmse_por_origen.csv")
    horizonte, mes, resumen = {}, {}, []
    for nombre, guardados in PRONOSTICOS.items():
        errores = pd.concat([(p - serie.loc[p.index]).rename("error").to_frame()
                             .assign(h=range(1, len(p) + 1), mes=p.index.month)
                             for p in guardados])
        horizonte[nombre] = errores.groupby("h")["error"].apply(lambda e: e.abs().mean())
        mes[nombre] = errores.groupby("mes")["error"].apply(lambda e: e.abs().mean())
        resumen.append({"modelo": nombre, "sesgo": round(errores["error"].mean(), 1),
                        "peor_mes": int(mes[nombre].idxmax()),
                        "MAE_peor_mes": round(mes[nombre].max(), 1),
                        "mejor_mes": int(mes[nombre].idxmin()),
                        "MAE_mejor_mes": round(mes[nombre].min(), 1),
                        "MAE_h1": round(horizonte[nombre].loc[1], 1),
                        "MAE_h12": round(horizonte[nombre].loc[12], 1),
                        "peor_origen": POR_ORIGEN[nombre].idxmax().strftime("%Y-%m"),
                        "RMSE_peor_origen": round(POR_ORIGEN[nombre].max(), 1)})
    pd.DataFrame(horizonte).round(1).to_csv(RESULTADOS / "error_por_horizonte.csv")
    pd.DataFrame(mes).round(1).to_csv(RESULTADOS / "error_por_mes.csv")
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(RESULTADOS / "errores.csv", index=False)
    print("  Dónde falla cada modelo (error = pronóstico - real, en GWh):")
    print("  " + resumen.to_string(index=False).replace("\n", "\n  "))

    figura, ejes = plt.subplots(1, 3, figsize=(13, 3.6))
    for nombre in destacados:
        ejes[0].plot(POR_ORIGEN[nombre].index, POR_ORIGEN[nombre].to_numpy(), marker="o",
                     markersize=3, label=nombre)
        ejes[1].plot(horizonte[nombre].index, horizonte[nombre].to_numpy(), marker="o", markersize=3)
        ejes[2].plot(mes[nombre].index, mes[nombre].to_numpy(), marker="o", markersize=3)
    ejes[0].set_title("RMSE por origen")
    ejes[0].tick_params(axis="x", labelrotation=45)
    ejes[1].set_title("Error absoluto medio por horizonte")
    ejes[1].set_xlabel("meses adelante")
    ejes[2].set_title("Error absoluto medio por mes del año")
    ejes[2].set_xlabel("mes pronosticado")
    ejes[2].set_xticks(range(1, 13))
    for eje in ejes:
        eje.set_ylabel("GWh")
        eje.set_ylim(bottom=0)
    ejes[0].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(FIGURAS / "errores.png", dpi=120)
    plt.close(figura)


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
        print("  coeficientes en el último origen:",
              {k: round(float(v), 3) for k, v in ULTIMO["modelo"].params.items()})
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        comparar_objetivos(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    filas_tabla = len(filas_completas(serie.iloc[:-12])[0]) if tabla_lista else 0
    print(f"  En el último origen hay {len(serie) - 12} meses de entrenamiento y "
          f"{filas_tabla} filas completas de la tabla.")
    destacados = [nombre for nombre in POR_ORIGEN
                  if any(nombre.startswith(elegido) for elegido in elegidos_para_figura)]
    analizar_errores(serie, ["estacional ingenuo"] + destacados)
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()

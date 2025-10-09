from pmdarima import auto_arima

#datos = [100, 200, 50, 700, 20, 30, 200, 300, 500, 1200, 600, 400, 0, 100, 120]
#datos = datos * 3

import pandas as pd
df = pd.read_csv("serie_tiempo.csv", parse_dates=["fecha"])
df = df.sort_values("fecha")     # Aplica si los registros no estan ordenados
datos = df["ventas"].values

modelo = auto_arima(datos, seasonal=False, trace=True)
print(modelo.summary())
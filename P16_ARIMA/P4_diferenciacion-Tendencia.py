import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

df = pd.read_csv("serie_tiempo.csv", parse_dates=["fecha"])
df = df.sort_values("fecha")     # Aplica si los registros no estan ordenados
y = df["ventas"].values

# Prueba de ADF en la serie original
resultado = adfuller(y)
print(f"p-valor serie original: {resultado[1]}")  # Si p > 0.05, diferenciamos

# Primera diferenciación (d = 1)
datos_d1 = np.diff(y) #y.diff(1)
resultado_d1 = adfuller(datos_d1)
print(f"p-valor con d=1: {resultado_d1[1]}")  # Si sigue p > 0.05, diferenciamos otra vez

# Segunda diferenciación (d = 2)
datos_d2 = np.diff(datos_d1) #datos_d1.diff(1)
resultado_d2 = adfuller(datos_d2)
print(f"p-valor con d=2: {resultado_d2[1]}")  # Si p < 0.05, la serie ya es estacionaria

# Graficamos las series
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(y, marker='o')
plt.title("Serie original")

plt.subplot(1,3,2)
plt.plot(datos_d1, marker='o')
plt.title("Serie diferenciada (d=1)")

plt.subplot(1,3,3)
plt.plot(datos_d2, marker='o')
plt.title("Serie diferenciada (d=2)")

plt.show()

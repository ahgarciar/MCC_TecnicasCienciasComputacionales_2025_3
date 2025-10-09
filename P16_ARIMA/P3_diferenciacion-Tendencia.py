
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Datos originales
datos = [100, 200, 50, 700, 20, 30, 200, 300, 500, 1200, 600, 400, 0, 100, 120]
datos = datos * 3

# Prueba de ADF en la serie original
resultado = adfuller(datos)
print(f"p-valor serie original: {resultado[1]}")  # Si p > 0.05, diferenciamos

# Primera diferenciación (d = 1)
datos_d1 = np.diff(datos)
resultado_d1 = adfuller(datos_d1)
print(f"p-valor con d=1: {resultado_d1[1]}")  # Si sigue p > 0.05, diferenciamos otra vez

# Segunda diferenciación (d = 2)
datos_d2 = np.diff(datos_d1)
resultado_d2 = adfuller(datos_d2)
print(f"p-valor con d=2: {resultado_d2[1]}")  # Si p < 0.05, la serie ya es estacionaria

# Graficamos las series
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(datos, marker='o')
plt.title("Serie original")

plt.subplot(1,3,2)
plt.plot(datos_d1, marker='o')
plt.title("Serie diferenciada (d=1)")

plt.subplot(1,3,3)
plt.plot(datos_d2, marker='o')
plt.title("Serie diferenciada (d=2)")

plt.show()

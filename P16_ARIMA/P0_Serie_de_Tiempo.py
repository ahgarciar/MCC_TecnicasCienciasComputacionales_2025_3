import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
anios = 5
meses = anios * 12

fechas = pd.date_range(start="2020-01-01", periods=meses, freq="MS")

tendencia = np.linspace(100, 300, meses)             # Crecimiento de ventas con el tiempo
estacionalidad = 30 * np.sin(2*np.pi*np.arange(meses)/12)  # Picos en ciertas épocas (ej. diciembre)
ruido = np.random.normal(0, 15, meses)               # Variación aleatoria (inesperado)

# Serie = tendencia + estacionalidad + ruido
ventas = tendencia + estacionalidad + ruido

serie = pd.DataFrame({"fecha": fechas, "ventas": ventas})

plt.figure(figsize=(9,4))
plt.plot(serie["fecha"], serie["ventas"], marker="o")
plt.title("Ventas mensuales")
plt.xlabel("Tiempo")
plt.ylabel("Unidades")
plt.grid(True)
plt.tight_layout()
plt.show()

serie.to_csv("serie_tiempo.csv", index=None)

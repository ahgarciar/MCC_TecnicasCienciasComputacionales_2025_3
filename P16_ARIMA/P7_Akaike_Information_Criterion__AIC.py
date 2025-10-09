from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("serie_tiempo.csv", parse_dates=["fecha"])
df = df.sort_values("fecha")     # Aplica si los registros no estan ordenados
y = df["ventas"].values

# Primera diferenciación (d = 1)
datos = np.diff(y) #y.diff(1)

import warnings
warnings.simplefilter("ignore")

d = 1 #serie estacionaria
p_values = [i for i in range(0, 17, 1)]
q_values = [i for i in range(0, 17, 1)]


best_aic = 999999  ##EN EL AIC, mientras mas bajo, mejor
best_model = None
best_order = None

for p in p_values:
    for q in q_values:
        try:
            model = ARIMA(datos, order=(p, d, q)).fit()
            aic = model.aic
            print("Parametros-> p: ",p, "   d: ", d, "  q: ", q, "      AIC:", aic)

            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_order = (p, d, q)
        except:
            print("Parametros-> p: ", p, "   d: ", d, "  q: ", q, "  COMBINACION NO VALIDA")


print(f"\nMejor modelo: ARIMA{best_order} con AIC = {best_aic}")

# Mejor modelo vs Datos Reales
plt.figure(figsize=(10, 5))
plt.plot(datos, label="Datos reales", color = "blue")
plt.plot(best_model.fittedvalues, label="Datos de ARIMA", linestyle="dashed", color = "green")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.simplefilter("ignore")

datos = [100, 200, 50, 700, 20, 30, 200, 300, 500, 1200, 600, 400, 0, 100, 120]
datos = datos * 3

d = 0 #serie estacionaria
p_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]


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

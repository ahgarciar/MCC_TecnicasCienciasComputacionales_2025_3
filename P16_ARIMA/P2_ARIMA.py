import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

df = pd.read_csv("serie_tiempo.csv", parse_dates=["fecha"])
df = df.sort_values("fecha")     # Aplica si los registros no estan ordenados
y = df["ventas"].values

modelo = ARIMA(y, order=(9,1,15))  # ARIMA(p,d,q)
ajuste = modelo.fit()

pasos_a_pronosticar = 10
pron = ajuste.forecast(steps=pasos_a_pronosticar)

#informacion del modelo entrenado (ajustado)
print(ajuste.summary())

print(f"Pronósticos:")
for i in range(pasos_a_pronosticar):
    print(f"Mes {i+1}: {pron[i]}")
    i += 1


# Agrega los pronosticos al final del grafico
fechas_fut = pd.date_range(df["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
                           periods=pasos_a_pronosticar, freq="MS")
plt.figure(figsize=(9,4))
plt.plot(df["fecha"], df["ventas"], label="Observado")
plt.plot(fechas_fut, pron, marker="o", label="Pronóstico")
plt.title("Observado vs Pronóstico ARIMA")
plt.xlabel("Tiempo"); plt.ylabel("Unidades vendidas")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
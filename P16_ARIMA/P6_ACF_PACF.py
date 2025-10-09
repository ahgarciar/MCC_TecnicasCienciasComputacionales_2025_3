
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("serie_tiempo.csv", parse_dates=["fecha"])
df = df.sort_values("fecha")     # Aplica si los registros no estan ordenados
y = df["ventas"].values

# Primera diferenciación (d = 1)
datos = np.diff(y) #y.diff(1)

plt.figure(figsize=(12,5))

# Creacion de ejes
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

#Para definir los LAGs se tiene que tomar en cuenta el tamano de la serie
# En ese sentido, para este ejemplo se tiene el maximo de lags es 6:
#Referencia: ValueError: Can only compute partial correlations for lags up to 50% of the sample size. The requested nlags 10 must be < 6.

# Grafica ACF (identifica q)
plot_acf(datos, ax=ax1) #lags=10)

# Grafica PACF (identifica p)
plot_pacf(datos, ax=ax2) #lags=10)


plt.show()

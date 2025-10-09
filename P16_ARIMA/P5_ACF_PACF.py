
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

datos = [100, 200, 50, 700, 20, 30, 200, 300, 500, 1200, 600, 400, 0, 100, 120]
datos = datos * 3

datos = np.array(datos) ##ARREGLO NOMPY

datos_d1 = np.diff(datos)
datos_d2 = np.diff(datos_d1)  ##d = 2 ... es la que permition lograr la serie estacionaria

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


from statsmodels.tsa.arima.model import ARIMA

datos = [100, 200, 50, 700, 20, 30, 200, 300, 500, 1200, 600, 400, 0, 100, 120]
datos = datos * 3

modelo = ARIMA(datos, order=(2,1,1))  # ARIMA(p,d,q)
ajuste = modelo.fit()

pronostico = ajuste.forecast(steps=1)  # Un paso adelante

print("Pronóstico:", pronostico[0])

#S la series es NO ESTAACIONARIA, aparecera esta advertencia:
# UserWarning: Non-stationary starting autoregressive parameters found
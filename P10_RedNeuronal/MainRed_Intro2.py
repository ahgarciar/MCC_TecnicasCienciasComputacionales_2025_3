from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load

from P05_KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("funcion.csv")

X = instancia.iloc[:,:-1]
y = pd.DataFrame(instancia.iloc[:,-1])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=7) #70/30
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=7) #15/15

#######################################################################################################################
model_loaded = load_model("modelo.keras")
x_scaler_loaded = load("x_scaler.joblib")
y_scaler_loaded = load("y_scaler.joblib")

X_test_s_loaded = x_scaler_loaded.transform(X_test)

y_pred_s_loaded = model_loaded.predict(X_test_s_loaded, verbose=0) # Devuelve a "y" estandarizada
y_pred_loaded = y_scaler_loaded.inverse_transform(y_pred_s_loaded) #pasa a escala original
y_test_arr = np.array(y_test).reshape(-1, 1) #redimensiona a 2D

# Métricas en unidades originales
mae_loaded = np.mean(np.abs(y_pred_loaded - y_test_arr))
mse_loaded = np.mean((y_pred_loaded - y_test_arr)**2)
print(f"Test MSE: {mse_loaded:.4f}, MAE: {mae_loaded:.4f}")

import matplotlib.pyplot as plt

y_true = y_test_arr.ravel() #flatten
y_pred  = y_pred_loaded.ravel() #flatten

# Ordenacion de "y" para que la curva tenga sentido visual
order = np.argsort(y_true) #devuelve indices que deberia tener el arreglo

plt.figure(figsize=(8,5))
plt.plot(y_true[order], label="y real", linewidth=2)
plt.plot(y_pred[order], label="y predicha", linewidth=2, alpha=0.8)
plt.xlabel("Índice ordenado por y real")
plt.ylabel("Valor")
plt.title("Comparación: y real vs y predicha (test)")
plt.legend()
plt.tight_layout()
plt.show()


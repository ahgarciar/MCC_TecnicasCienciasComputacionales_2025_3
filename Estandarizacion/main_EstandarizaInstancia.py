import numpy as np
import pandas as pd

from KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("../Archivos/iris/iris.csv")
#print(instancia)

X = instancia.iloc[:,:-1]
Y = pd.DataFrame(instancia.iloc[:,-1])
print(X)

Xarray = X.to_numpy()

from sklearn.preprocessing import StandardScaler as escalador
#from sklearn.preprocessing import MinMaxScaler as escalador

Xstd = scaler = escalador().fit_transform(Xarray)

Xstd = pd.DataFrame(data=Xstd, columns=[X.columns])

Xstd["class"] = Y

Xstd.to_csv("../Archivos/iris/iris_estandarizada.csv", index=None)
#Xstd.to_csv("../Archivos/iris/iris_normalizada.csv", index=None)

print()

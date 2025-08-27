import CargaInstancia
import Knn_algoritmo

instancia = CargaInstancia.cargarInstancia("InstanciaTennis.csv")

import math as m
k = m.sqrt(len(instancia))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))

Knn_algoritmo.exec(instancia, k)

#for i in range(1,len(instancia)):
#    Knn_algoritmo.exec(instancia, i)


import pandas as pd

import CargaInstancia
import Knn_algoritmo

def split_instance(instance, porcentajeEntrenamiento):
    totRegEntrenamiento = int(len(instance)*porcentajeEntrenamiento)
    totRegPrueba = len(instance)-totRegEntrenamiento

    import random
    indices = [i for i in range(len(instance))]
    random.shuffle(indices)

    entrenamiento = pd.DataFrame([])
    for i in range(totRegEntrenamiento):
        registro = instance.loc[[indices[i]]]
        entrenamiento = pd.concat([entrenamiento, registro])
        #print()

    prueba = pd.DataFrame([])
    for i in range(totRegPrueba):
        registro = instance.loc[[indices[i+totRegEntrenamiento]]]
        prueba = pd.concat([prueba, registro])

    print()

instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennis.csv")

porc_Entrenamiento = 0.6
entrenamiento, prueba = split_instance(instancia, porc_Entrenamiento)

import math as m
k = m.sqrt(len(instancia))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))

registro = [9, "Sunny", "Cool", "Normal", "Weak", "Yes"]

respuesta = Knn_algoritmo.exec(instancia, registro, k)

print(respuesta)

#for i in range(1,len(instancia)):
#    Knn_algoritmo.exec(instancia, i)


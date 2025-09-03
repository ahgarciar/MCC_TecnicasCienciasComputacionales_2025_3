import pandas as pd
from tensorflow.python.ops.metrics_impl import precision

import CargaInstancia
import Knn_algoritmo

def split_instance(instance, porcentajeEntrenamiento):
    #instance = instance.drop("Day", axis=1)
    instance.drop("Day", axis=1, inplace=True)
    
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

    #print()
    entrenamiento.reset_index(inplace=True, drop=True)
    prueba.reset_index(inplace=True, drop=True)
    
    return entrenamiento, prueba

def ejecutarKnn(entrenamiento, prueba, k):
    respuestas =[]
    for i in range(len(prueba)):
        registro = list(prueba.loc[i])
        respuesta = Knn_algoritmo.exec(entrenamiento, registro, k)
        respuestas.append(respuesta)
    return respuestas



instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennis.csv")

porc_Entrenamiento = 0.6
entrenamiento, prueba = split_instance(instancia, porc_Entrenamiento)

import math as m
k = m.sqrt(len(entrenamiento))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))

respuestas = ejecutarKnn(entrenamiento, prueba, k)
print(respuestas)

respCorrecta = list(prueba["Play Tennis"])
print(respCorrecta)

TP = 0
FN = 0
FP = 0
TN = 0


for i in range(len(respuestas)):
    if respCorrecta[i] == "Yes": ##REAL
        if respuestas[i] == "Yes": #KNN
            TP += 1
        else: #No
            FN += 1
    else: #No  #REAL
        if respuestas[i] == "Yes": ## KN
            FP += 1
        else: #No
            TN += 1

eficiencia = (TP + TN) /(TP + FN + FP + TN)
precision = TP / (TP + FP)
recall = TP/(TP + FN)
f1_score = 2 * ((precision*recall)/(precision+recall))

print("Tot Correctas: " + str(TP + TN))
print("Tot Pruebas: " + str(len(prueba)))

print("Eficiencia: " + str(eficiencia))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1-Score: " + str(f1_score))

#for i in range(1,len(instancia)):
#    Knn_algoritmo.exec(instancia, i)


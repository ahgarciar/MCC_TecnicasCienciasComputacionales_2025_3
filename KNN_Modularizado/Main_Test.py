import CargaInstancia
import KNN_Algoritmo
import SplitInstacia
import MatrizConfusion

instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennis.csv")

porc_Entrenamiento = 0.6
entrenamiento, prueba = SplitInstacia.split_instance(instancia, porc_Entrenamiento)

################################################################################
##Calcula valor de K
import math as m
k = m.sqrt(len(entrenamiento))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))
################################################################################
##get respuestas por algoritmo
respKnn = KNN_Algoritmo.probarKnn(entrenamiento, prueba, k)
print(respKnn)
##get respuestas reales
respCorrectas = list(prueba["Play Tennis"])
print(respCorrectas)
################################################################################
##calcula matriz de confusion
MatrizConfusion.exec(respCorrectas, respKnn)

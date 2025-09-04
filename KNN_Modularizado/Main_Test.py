import CargaInstancia
import KNN_Algoritmo
import SplitInstacia
import MatrizConfusion

instancia = None

id = 0

match(id):
    case 0:
        instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennis.csv")
        instancia.drop("Day", axis=1, inplace=True)
    case 1:
        instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennisCodLabel.csv")
    case 2:
        instancia = CargaInstancia.cargarInstancia("../Archivos/InstanciaTennisCodOneHotVector.csv")


porc_Entrenamiento = 0.6
entrenamiento, prueba = SplitInstacia.split_instance(instancia, porc_Entrenamiento)

################################################################################
##Calcula valor de K
import math as m
k = m.sqrt(len(entrenamiento))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))
################################################################################
ejecuciones = 0
tot_ejecuciones = 30
resultados = []
while ejecuciones<tot_ejecuciones:
    ################################################################################
    ##get respuestas por algoritmo
    respKnn = KNN_Algoritmo.probarKnn(entrenamiento, prueba, k, tipoMetrica=0)
    #print(respKnn)
    ##get respuestas reales
    respCorrectas = list(prueba["Play Tennis"])
    #print(respCorrectas)
    ################################################################################
    ##calcula matriz de confusion
    resultado = MatrizConfusion.exec(respCorrectas, respKnn)
    resultado = round(resultado[0], 2)
    resultados.append(resultado)
    ################################################################################
    entrenamiento, prueba = SplitInstacia.split_instance(instancia, porc_Entrenamiento)
    ejecuciones +=1
print(resultados)



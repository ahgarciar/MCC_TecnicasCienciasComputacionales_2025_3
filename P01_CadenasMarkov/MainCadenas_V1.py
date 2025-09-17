
archivo = open("MatrizDeEstados.csv")
content = archivo.readlines()
print(content)
instancia = [i.split(",") for i in content]
print(instancia)
matriz = [list(map(float, i)) for i in instancia]
print(matriz)
#for fila in matriz:
#    print(fila)
    
import numpy as np
matriz = np.array(matriz)
print(matriz)
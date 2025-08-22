
def hamming(vector1, vector2):
    distancia = 0
    for i in range(1, len(vector1)-1):
        if vector1[i] != vector2[i]:
            distancia+=1
    return distancia


import pandas as pd
instancia = pd.read_csv("InstanciaTennis.csv")

#print(instancia)
#comparacion del registro 1 con todos los demas registros
print("Registro a comparar:")
registro_comp = list(instancia.loc[0])
print(registro_comp)
print(" resultado de la comparación:..")
for i in range(1,len(instancia)):
    registro = list(instancia.loc[i])
    dist = hamming(registro_comp, registro)
    print(str(registro) + " Distancia: " + str(dist))

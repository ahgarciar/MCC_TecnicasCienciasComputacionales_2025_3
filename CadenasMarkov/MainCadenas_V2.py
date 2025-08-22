import numpy as np
import pandas as pd
matriz =  pd.read_csv('MatrizDeEstados.csv', header=None)
print(matriz.head())
matriz = matriz.to_numpy()
print(matriz)
p0 = [0.3, 0.6,	0.1]
p0 = np.array(p0)
p1 = p0.dot(matriz) #.dot = multiplicacion de matrices <<<---
print(p1)
p2 = p1.dot(matriz)
print(p2)

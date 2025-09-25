import math
import numpy as np

#producto punto
def dot(a, b):
    s = 0.0
    for k in range(len(a)):
        s += float(a[k]) * float(b[k])
    return s

#distancia euclidiana al cuadrado (norma 2, al cuadrado)
def dist_euclidana_al_cuadrado(a, b):
    s = 0.0
    for k in range(len(a)):
        diff = float(a[k]) - float(b[k])
        s += diff * diff
    return s

#linear
def linear_kernel(X):
    n = len(X)
    K = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = dot(X[i], X[j])
    return np.array(K, dtype=float)

def rbf_kernel(X, sigma=1.5):
    gamma = 1.0 / (2.0 * sigma * sigma)
    n = len(X)
    K = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d2 = dist_euclidana_al_cuadrado(X[i], X[j])
            K[i][j] = math.exp(-gamma * d2)
    return np.array(K, dtype=float), gamma

def poly_kernel(X, degree=3, gamma=1.0, coef0=1.0):
    n = len(X)
    K = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dp = dot(X[i], X[j])
            K[i][j] = (gamma * dp + coef0) ** degree
    return np.array(K, dtype=float)

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 10
    datos_por_clase = int(n/2)
    #puntos de la clase 0
    X0 = rng.normal(loc=[-5, -5], scale=1.0, size=(datos_por_clase, 2))
    #puntos de la clase 1
    X1 = rng.normal(loc=[ 5,  5], scale=1.0, size=(datos_por_clase, 2))

    X = np.vstack([X0, X1])

    linear = linear_kernel(X)
    rbf, gamma = rbf_kernel(X, sigma=1.5)
    poly = poly_kernel(X, degree=3, gamma=1.0, coef0=1.0)

    print("Matriz de kernel:")
    for fila in linear:
        for columna in fila:
            print(round(columna,2), end=" ")
        print()


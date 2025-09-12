import numpy as np
import pandas as pd

# Dominio: x1, x2 ~ Uniforme[-2, 2]
N = 20_000 #20mil valores para x1 y x2
rnd = np.random.default_rng(42) #generador de aletorios con semilla

#Distribución uniforme significa que todos los valores tienen la misma probabilidad de ocurrir
X = rnd.uniform(-5.0, 5.0, size=(N, 2))
x1, x2 = X[:, 0], X[:, 1]

def funcion(x1, x2):# y = f(x1,x2)
    return np.sin(3*x1) + 0.5*np.cos(5*x2) + 0.1*x1*x2 + 0.2*(x1**2) - 0.3*(x2**2)

y = funcion(x1, x2)

# Ruido para simular medición (sensor)
noise = rnd.normal(0, 0.1, size=N)
y = y + noise
y = y.reshape(-1, 1)

instancia = pd.DataFrame(np.concatenate([X, y], axis=1), columns=["X1", "X2", "Y"])

instancia.to_csv("funcion.csv", index=None)

print()
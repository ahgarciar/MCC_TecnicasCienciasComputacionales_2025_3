import random as rnd
rnd.seed(5)

tot_samples = 100
x1 = [rnd.randint(-10, 10) for i in range(tot_samples)]
x2 = [rnd.randint(-10, 10) for i in range(tot_samples)]

y = []
for i in range(tot_samples):
    valor = 3*x1[i]+ 6*x2[i]+12
    valor = valor + rnd.randint(-5,5) #ruido
    y.append(valor)


X = []
for i in range(tot_samples):
    X.append([1, x1[i], x2[i]])

import numpy as np
X = np.array(X)
y = np.array(y)

x1 = np.array(x1)
x2 = np.array(x2)

aux = X.T.dot(X)
aux = np.linalg.inv(aux)
aux = aux.dot(X.T)
b_estimada = aux.dot(y)

y_estimada = X.dot(b_estimada)

from P14_IntroduccionRegresion import MetricasError as calc

rmse = calc.rmse(y, y_estimada)

print("RMSE: ", rmse)


from matplotlib import pyplot as plt
x1_grid = np.linspace(x1.min(), x1.max(), 30)
x2_grid = np.linspace(x2.min(), x2.max(), 30)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)
Yg = b_estimada[0] + b_estimada[1] * X1g + b_estimada[2] * X2g

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y, alpha=0.7)
ax.plot_surface(X1g, X2g, Yg, rstride=1, cstride=1, alpha=0.3)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Regresión lineal 3D: puntos y plano ajustado")
plt.tight_layout()
plt.show()


print()
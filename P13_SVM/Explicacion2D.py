import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def entrenar_y_graficar(X, y, kernel, C=1.0, gamma="scale", degree=3):
    #ESCALA VALORES DE ENTRADA
    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X)

    #CREA Y ENTRENA EL MODELO
    modelo = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    modelo.fit(X_escalada, y)

    #FRONTERA DE DECISION DEL MODELO SVM (Permite visualizar lo que el modelo aprendio
    #RANGO DEL PLANO (EXTREMOS CON MARGEN (DE 0.5)
    x_min, x_max = X_escalada[:, 0].min() - 0.5, X_escalada[:, 0].max() + 0.5
    y_min, y_max = X_escalada[:, 1].min() - 0.5, X_escalada[:, 1].max() + 0.5
    #MALLA/GRILLA DE PUNTOS
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    #300 PUNTOS POR EJE (300 para "x" y 300 para "y"... Cada PAR sera un candidato para validar,es decir, comprobar que clase le asigna el MODELO

    #calcula el valor de la función de decisión para cada punto.
    # Si es positivo: el modelo cree que esta en la clase 1.
    # Si es negativo: el modelo cree que esta en la clase 0.
    # Si es cerca de 0: está en la frontera.
    Z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, (Z > 0).astype(int), alpha=0.2) #pinta areas en el plano
    #margenes: buscan asegurar que la separación entre clases sea lo más robusta posible
    CS = plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"])
    plt.clabel(CS, inline=True, fontsize=8)
    #plt.show()

    #s = tamano de cada punto
    plt.scatter(X_escalada[:, 0], X_escalada[:, 1], s=25, edgecolor="k")
    sv = modelo.support_vectors_ # puntos de datos criticos para la frontera de decision
    plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolor="k", linewidths=1.3, label="SV")

    plt.title(f"SVM kernel={kernel}")
    plt.xlabel("x (escalada)")
    plt.ylabel("y (escalada)")
    plt.legend(loc="best")
    plt.tight_layout()

    #plt.show()

    #print()


if __name__ == "__main__":
    rnd = np.random.default_rng(5)
    n = 240
    datos_por_clase = int(n/3)  #entre 2 porque son dos clases las que se usaran en este ejemplo
    #loc: Mean (“centre”) of the distribution.
    #scale: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    #size: Output shape.
    X0 = rnd.normal(loc=[-5, -5], scale=1.0, size=(datos_por_clase, 2)) #datos en el cuadrante IV
    #X1 = rnd.normal(loc=[ 5,  5], scale=1.0, size=(datos_por_clase, 2)) #datos en el cuadrante I
    X1 = rnd.normal(loc=[-3, -2], scale=1.0, size=(datos_por_clase, 2))  # datos en el cuadrante I
    X = np.vstack([X0, X1]) #atributos de la instancia
    y = np.array([0]*datos_por_clase + [1]*datos_por_clase) # clase ordinalizada para cada registro

    #datos iniciales
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X0[:, 0], X0[:, 1], c="blue", label="Clase 0", alpha=0.7)
    plt.scatter(X1[:, 0], X1[:, 1], c="red", label="Clase 1", alpha=0.7)
    plt.scatter(X2[:, 0], X2[:, 1], c="green", label="Clase 2", alpha=0.7)
    plt.title("Datos iniciales X0 y X1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
"""

    #C = tolerancia a errores
    # LINEAR (util si la separación es casi recta)
    entrenar_y_graficar(X, y, kernel="linear", C=1.0)

    # RBF (captura no linealidad)
    entrenar_y_graficar(X, y, kernel="rbf", C=1.0, gamma="scale")

    # POLY
    entrenar_y_graficar(X, y, kernel="poly", C=1.0, gamma="scale", degree=3)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)

nombres = {
    1: "Recta",
    2: "Parábola",
    3: "Cúbica",
    4: "Cuártica",
    5: "Quíntica"
}


for i, grado in enumerate(range(1, 6)):
    y = x**grado
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, label=f"$x^{grado}$", color="b")
    plt.title(f"Grado {grado} → {nombres[grado]}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout() #ajuste automatico entre subgraficas para evitar que se encimen
    plt.show()

"""
fig, axs = plt.subplots(5, 1, figsize=(6, 14))

for i, grado in enumerate(range(1, 6)):
    y = x**grado
    axs[i].plot(x, y, label=f"$x^{grado}$", color="b")
    axs[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axs[i].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axs[i].set_title(f"Grado {grado} → {nombres[grado]}")
    axs[i].legend(loc="upper left")
    axs[i].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout() #ajuste automatico entre subgraficas para evitar que se encimen
plt.show()
"""
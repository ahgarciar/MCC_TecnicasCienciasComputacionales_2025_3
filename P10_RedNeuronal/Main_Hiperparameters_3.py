import numpy as np
import pandas as pd
from keras import Sequential,layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_model(X_train, n_layers=2, units=128, dropout=0.1, lr=1e-3):
    model = Sequential([layers.Input(shape=(X_train.shape[1],))]) #entrada
    for _ in range(n_layers):
        model.add(layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear")) #1 salida = y
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def create_solution(rng):
    return {
        "n_layers": int(rng.integers(1, 4)),          # 1,2,3
        "units": int(rng.choice([64, 128, 256])),
        "dropout": float(rng.choice([0.0, 0.1, 0.2])),
        "lr": float(10 ** rng.uniform(-4, -2.5)),     # ~1e-4 a ~3e-3
        "batch_size": int(rng.choice([128, 256, 512])),
        "epochs": 200
    }

def exec(X_train_s, y_train_s, X_val_s, y_val_s, iteraciones=12, patience=15, seed=123):
    rng = np.random.default_rng(seed)
    resultados = []
    mejor = {"val_loss": np.inf, "params": None, "weights": None, "hist": None}

    t = 1
    while t < iteraciones+1:
        try:
            params = create_solution(rng)
            model = create_model(X_train_s, n_layers=params["n_layers"], units=params["units"], dropout=params["dropout"], lr=params["lr"])

            early = callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                            restore_best_weights=True, verbose=0)

            hist = model.fit(
                X_train_s, y_train_s,
                validation_data=(X_val_s, y_val_s),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
                callbacks=[early]
            )

            val_loss = float(np.min(hist.history["val_loss"]))
            val_mae  = float(np.min(hist.history.get("val_mae", [np.nan])))

            resultados.append({
                "intento": t, "val_loss": val_loss, "val_mae": val_mae, **params
            })

            print(f"[{t}/{iteraciones}] val_loss={val_loss:.5f} con {params}", end="")

            if val_loss < mejor["val_loss"]:
                mejor = {"val_loss": val_loss, "params": params, "weights": model.get_weights(), "hist": hist}
                print("\tNuevo mejor encontrado", end="")

            print()
            t += 1
        except Exception as error:
            print("Modelo no valido... se reintentará una nueva configuración...")

    ranking = pd.DataFrame(resultados).sort_values("val_loss").reset_index(drop=True)
    return mejor, ranking

from P05_KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("funcion.csv")

X = instancia.iloc[:,:-1]
y = pd.DataFrame(instancia.iloc[:,-1])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=7) #70/30
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=7) #15/15

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)   # facilita el calculo de la funcion de perdida (loss)

X_train_s = x_scaler.transform(X_train)
X_val_s   = x_scaler.transform(X_val)
X_test_s  = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)
y_val_s   = y_scaler.transform(y_val)
y_test_s  = y_scaler.transform(y_test)

mejor, ranking = exec(X_train_s, y_train_s, X_val_s, y_val_s, iteraciones=12, patience=12, seed=42)

print("\nTOP de configuraciones por val_loss:")
print(ranking.head(5))
print("\nMejor configuración encontrada:\n", mejor["params"], "\nval_loss:", mejor["val_loss"])


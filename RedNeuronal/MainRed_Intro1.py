import numpy as np
import pandas as pd
from keras import Sequential,layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from P7_Knn_conModularizado import LoadInstance

instancia = LoadInstance.load("funcion.csv")

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

def create_model():
    model = Sequential([
        layers.Input(shape=(2,)), #entrada -> x1, x2

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(1, activation="linear") #salida -> y
    ])
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

model = create_model()

early = callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=300,
    batch_size=256,
    callbacks=[early],
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test_s, y_test_s, verbose=0)
# Pasa a unidades originales para interpretacion de resultados
y_pred_s = model.predict(X_test_s, verbose=0) #escalado
y_pred = y_scaler.inverse_transform(y_pred_s) #original
mae_orig = np.mean(np.abs(y_pred - y_test))
mse_orig = np.mean((y_pred - y_test)**2)

print(f"Con datos Escalados: Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")
print(f"Con datos originales Test MSE: {mse_orig:.4f}, MAE: {mae_orig:.4f}")


import matplotlib.pyplot as plt
hist_df = pd.DataFrame(history.history)

plt.figure()
plt.plot(hist_df["loss"], label="train_loss")
plt.plot(hist_df["val_loss"], label="val_loss")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Comportamiento de Loss (MSE)")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(hist_df["mae"], label="train_mae")
plt.plot(hist_df["val_mae"], label="val_mae")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.title("Comportamiento de MAE")
plt.legend()
plt.tight_layout()

plt.show()


from joblib import dump
model.save("modelo.keras")
dump(x_scaler, "x_scaler.joblib")
dump(y_scaler, "y_scaler.joblib")

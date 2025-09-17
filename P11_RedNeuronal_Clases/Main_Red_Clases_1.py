import pandas as pd
from keras import Sequential, layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from joblib import dump

from P05_KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("../Archivos/iris/iris.csv")
#########################################################################################################
##SEPARA LA INSTANCIA
#copia para asegurar no alterar a la instancia original al hacer ajustes en X e y
X = instancia.iloc[:, :-1].copy()
y = instancia.iloc[:, -1].copy()   # nominales
#########################################################################################################
##CODIFICA LAS CLASES
encoder = LabelEncoder() #ordinalizacion (Label)
y_int = encoder.fit_transform(y)
#########################################################################################################
##CALCULA EL TTOTAL DE CLASES Y DE ATRIBUTOS
n_clases = len(encoder.classes_)
n_feats = X.shape[1]
#########################################################################################################
# Split 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_int, test_size=0.30, random_state=7, stratify=y_int #APLICA ESTRATIFICACION
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=7, stratify=y_temp #APLICA ESTRATIFICACION
)
#########################################################################################################
# Escalado de X
x_scaler = StandardScaler().fit(X_train)
X_train_s = x_scaler.transform(X_train)
X_val_s   = x_scaler.transform(X_val)
X_test_s  = x_scaler.transform(X_test)
#########################################################################################################
# CREACION DEL MODELO
def create_model(input_dim, n_classes):
    model = Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(n_classes, activation="softmax")  # IDEAL PARA MULTICLASES
    ])
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",   # PARA VARIABLES CATEGORICAS
        metrics=["accuracy"]
    )
    return model

model = create_model(n_feats, n_clases)

early = callbacks.EarlyStopping(
    patience=20, restore_best_weights=True, monitor="val_loss"
)

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=300,
    batch_size=256,
    callbacks=[early],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"Test  - Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

y_probabilities = model.predict(X_test_s, verbose=0)
y_pred  = y_probabilities.argmax(axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm_df)


hist_df = pd.DataFrame(history.history)

plt.figure()
plt.plot(hist_df["loss"], label="train_loss")
plt.plot(hist_df["val_loss"], label="val_loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Comportamiento de Loss")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(hist_df["accuracy"], label="train_acc")
plt.plot(hist_df["val_accuracy"], label="val_acc")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Comportamiento de Accuracy")
plt.legend()
plt.tight_layout()

plt.show()

model.save("modelo_clasificacion.keras")
dump(x_scaler, "x_scaler.joblib")
# Guardar el LabelEncoder para reusar el mapeo de clases
dump(encoder, "label_encoder.joblib")

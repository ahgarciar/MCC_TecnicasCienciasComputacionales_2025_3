import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from joblib import dump

# CARGA DE INSTANCIA
from P05_KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("../Archivos/iris/iris.csv")

# SEPARA LA INSTANCIA
X = instancia.iloc[:, :-1].copy()
y = instancia.iloc[:, -1].copy()   # nominal

# CODIFICA LAS CLASES
encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

n_clases = len(encoder.classes_)
n_feats = X.shape[1]
print(f"n_feats={n_feats} | n_clases={n_clases} | clases={list(encoder.classes_)}")

# SPLIT 70/15/15 con estratificación
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_int, test_size=0.30, random_state=7, stratify=y_int
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=7, stratify=y_temp
)

# ESCALADO (con TRAIN)
x_scaler = StandardScaler().fit(X_train)
X_train_s = x_scaler.transform(X_train)
X_val_s   = x_scaler.transform(X_val)
X_test_s  = x_scaler.transform(X_test)

# MODELO SVM
# - kernel RBF -  para problemas no lineales
params_grid = [
    # Kernel lineal
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1, 10, 100]
    },
    # Kernel RBF (gaussiano)
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001]
    },
    # Kernel polinómico
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1, 0.1, 0.01],
        "degree": [2, 3, 4, 5],  # típicamente 2 o 3,
        "coef0": [0.0, 0.5, 1.0]  # término independiente del polinomio
    }
]

params= params_grid[1] #rbf
# - probability=True para obtener predict_probabilities
svm_model = SVC(kernel=params["kernel"][0], C=params["C"][1], gamma=params["gamma"][0], probability=True, random_state=7)

# Entrenamiento con TRAIN
svm_model.fit(X_train_s, y_train)

# EVALUACIÓN EN VALIDACIÓN
val_pred = svm_model.predict(X_val_s)
val_acc  = accuracy_score(y_val, val_pred)
print(f"\nValidación - Accuracy: {val_acc:.4f}")

# REENTRENAMIENTO con TRAIN + VAL (recomendado antes de TEST)
X_trainval = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])

svm_final = SVC(kernel=params["kernel"][0], C=params["C"][1], gamma=params["gamma"][0], probability=True, random_state=7)
svm_final.fit(X_trainval, y_trainval)

# EVALUACIÓN FINAL EN TEST
y_prob = svm_final.predict_proba(X_test_s) ##probabilities
y_pred = svm_final.predict(X_test_s)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest - Accuracy: {test_acc:.4f}")

print("\nClassification report (TEST):")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm_df)

dump(svm_final, "svm_model.joblib")
dump(x_scaler,  "x_scaler.joblib")
dump(encoder,   "label_encoder.joblib")

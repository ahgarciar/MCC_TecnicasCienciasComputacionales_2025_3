import keras.utils
import random as rnd
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from AG_LSTM import CustomNormalization, calc_rmse, train_val_test_split, create_supervised_dataset

# Escalando cada característica por separado
scalers = dict()  # CREAR GLOBAL
data = dict()
normalization_layer = None
# Definimos los valores globales para la multiplicación


def load_instance(instance_name=''):
    global normalization_layer

    base_folder = 'instancias'

    if os.path.isfile(os.path.join(base_folder, instance_name)) and instance_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(base_folder, instance_name))

        ###################################################################
        # Convierte la columna 'Fecha' a tipo datetime
        df['date'] = pd.to_datetime(df['Fecha'])
        # Obtener la semana, mes y año de "DATE"
        df['week'] = df['date'].dt.isocalendar().week  # 1 al 52 (53) #semana del año
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        ###################################################################
        df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
        ###################################################################
        instance = df[['year', 'sin_week', 'cos_week', 'Evento', 'Ventas']]
        instance_std = pd.DataFrame([])

        for index in range(len(instance.columns)):
            column = instance.columns[index]
            scaler = StandardScaler()
            instance_std[column] = pd.DataFrame(scaler.fit_transform(instance[[column]]), columns=[column])
            scalers.update({column: scaler})

        mean = scalers['Ventas'].mean_
        std = scalers['Ventas'].scale_

        normalization_layer = CustomNormalization(mean, std)

        return instance_std


if __name__ == '__main__':

    tf.config.experimental.enable_op_determinism()  # LLLLLLLLLL

    rnd.seed(5)
    np.random.seed(5)
    tf.random.set_seed(5)
    keras.utils.set_random_seed(5)

    tf.keras.backend.clear_session()

    instance = load_instance("Instancia_Producto1.csv")

    model = load_model('Mejor_Modelo/modelo_mejora_4.keras', custom_objects={
        'CustomNormalization': CustomNormalization,
        'calc_rmse': calc_rmse
    })

    tr, vl, ts = train_val_test_split(instance)
    # print(f'Tamaño set de entrenamiento: {tr.shape}')
    # print(f'Tamaño set de validacion: {vl.shape}')
    # print(f'Tamaño set de prueba: {ts.shape}')

    # Definición de los hiperparámetros INPUT_LENGTH y OUTPUT_LENGTH
    INPUT_LENGTH = 26  # semanas de entrada
    OUTPUT_LENGTH = 1  # semana futura

    # Datasets supervisados para entrenamiento (x_tr, y_tr), validación (x_vl, y_vl) y prueba (x_ts, y_ts)
    x_tr, y_tr = create_supervised_dataset(tr.values, INPUT_LENGTH, OUTPUT_LENGTH)
    x_vl, y_vl = create_supervised_dataset(vl.values, INPUT_LENGTH, OUTPUT_LENGTH)
    x_ts, y_ts = create_supervised_dataset(ts.values, INPUT_LENGTH, OUTPUT_LENGTH)

    print('Tamaños entrada (BATCHES x INPUT_LENGTH x FEATURES) y de salida (BATCHES x OUTPUT_LENGTH x FEATURES)')
    print(f'Set de entrenamiento - x_tr: {x_tr.shape}, y_tr: {y_tr.shape}')
    print(f'Set de validación - x_vl: {x_vl.shape}, y_vl: {y_vl.shape}')
    print(f'Set de prueba - x_ts: {x_ts.shape}, y_ts: {y_ts.shape}')

    data = {
        'x_tr': x_tr, 'y_tr': y_tr,
        'x_vl': x_vl, 'y_vl': y_vl,
        'x_ts': x_ts, 'y_ts': y_ts,
    }

    print()

    y_pred = model(data['x_ts'])
    y_pred = y_pred.numpy()
    y_true = data['y_ts'].reshape(-1, 1)

    import MetricasDeError as error
    rmse_manual = error.calcRMSE(y_true, y_pred)
    mape_manual = error.calcMAPE(y_true, y_pred)

    #valores originales
    #y_pred = normalization_layer.inverse(y_pred)
    #y_pred = np.round(y_pred)


    rmse_tr = model.evaluate(x=x_tr, y=y_tr, verbose=0)
    rmse_vl = model.evaluate(x=x_vl, y=y_vl, verbose=0)
    rmse_ts = model.evaluate(x=x_ts, y=y_ts, verbose=0)

    rmse_manual2 = error.calcRMSE(y_true, y_pred)
    mape_manual2 = error.calcMAPE(y_true, y_pred)

    print(f"RMSE train: \t{rmse_tr:.3f}")
    print(f"RMSE val: \t{rmse_vl:.6f}")
    print(f"RMSE test: \t{rmse_ts:.3f}")
    print(f"Manual RMSE test: \t{rmse_manual:.3f}")
    print(f"Manual RMSE2 test: \t{rmse_manual2:.3f}")
    print(f"Manual MAPE test: \t{mape_manual:.3f}")
    print(f"Manual MAPE2 test: \t{mape_manual2:.3f}")

    # APLICADO PARA LAS PRIMERAS GRAFICAS
    x = [(i + 1) for i in range(len(y_pred))]
    plt.figure(figsize=(20, 7))
    plt.plot(x, y_pred, linewidth=2, marker="s", color="blue", label="Forecast")
    plt.plot(x, y_true, linestyle="--", linewidth=2, marker="o", color="red", label="Actual")

    plt.title("Product 1: Dining room", weight='bold', size="16")

    plt.ylabel("Sales", weight='bold', size="14")
    plt.xlabel("Week", weight='bold', size="14")
    #plt.ylim([0, 1])
    #plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 100])
    #plt.yticks([0, , 30, 40, 50, 60, 70, 80, 100])
    plt.xticks(x)

    from matplotlib.font_manager import FontProperties
    legend_properties = FontProperties(size=20, weight='bold')
    plt.legend(loc="upper left", prop=legend_properties)

    plt.grid(True)
    plt.show()

    #nombre_figura = "imagen.png"

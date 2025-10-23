import keras.utils
import random as rnd
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from main_LSTM_sandra import CustomNormalization, calc_rmse, train_val_test_split, create_supervised_dataset

# Escalando cada característica por separado
scalers = dict()  # CREAR GLOBAL
data = dict()
normalization_layer = None
# Definimos los valores globales para la multiplicación


def load_instance(instance_name=''):
    global normalization_layer

    base_folder = 'instancias'
    path = os.path.join(base_folder, instance_name)
    if os.path.isfile(path) and instance_name.endswith('.csv'):
        df = pd.read_csv(path)

        df['date'] = pd.to_datetime(
            dict(year=df['YEAR'], month=df['MO'], day=df['DY'], hour=df['HR'])
        )
        df = df.sort_values('date').reset_index(drop=True)

        # hora del día (0-23)
        df['hour'] = df['date'].dt.hour
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        # día de la semana (0=lunes,...,6=domingo)
        df['dow'] = df['date'].dt.weekday
        df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7)

        # día del año (1-365/366)
        df['doy'] = df['date'].dt.dayofyear
        df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365)
        df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365)

        # direcciones de viento a sin/cos para evitar salto de 359° a 0°
        for col in ['WD10M', 'WD50M']:
            if col in df.columns:
                rad = np.deg2rad(df[col].astype(float))
                df[f'{col.lower()}_sin'] = np.sin(rad)
                df[f'{col.lower()}_cos'] = np.cos(rad)

        # primero las features y al último T2M (variable a pronosticar)
        feature_cols = []
        # variables temporales
        feature_cols += ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy']
        # meteorológicas escalares
        for c in ['WS10M', 'PS', 'WS50M']:
            if c in df.columns:
                feature_cols.append(c)
        # componentes de dirección ya transformadas
        for c in ['wd10m_sin', 'wd10m_cos', 'wd50m_sin', 'wd50m_cos']:
            if c in df.columns:
                feature_cols.append(c)

        target_col = 'T2M'

        instance = df[feature_cols + [target_col]].copy()

        # escalado de cada columna por separado
        instance_std = pd.DataFrame(index=instance.index)
        for column in instance.columns:
            scaler = StandardScaler()
            instance_std[column] = scaler.fit_transform(instance[[column]])
            scalers.update({column: scaler})

        instance_std = instance_std.astype(np.float32)

        # estadísticas de la variable objetivo (T2M) para la capa CustomNormalization
        mean = scalers[target_col].mean_
        std = scalers[target_col].scale_
        normalization_layer = CustomNormalization(mean, std)

        del df, instance, feature_cols, path

        return instance_std


if __name__ == '__main__':

    tf.config.experimental.enable_op_determinism()  # LLLLLLLLLL
    tf.keras.backend.set_floatx('float32')

    rnd.seed(5)
    np.random.seed(5)
    tf.random.set_seed(5)
    keras.utils.set_random_seed(5)

    tf.keras.backend.clear_session()

    instance = load_instance("instancia_sandra.csv")

    model = load_model('Mejor_Modelo/modelo_solution_0.keras', custom_objects={
        'CustomNormalization': CustomNormalization,
        'calc_rmse': calc_rmse
    })

    tr, vl, ts = train_val_test_split(instance)
    # print(f'Tamaño set de entrenamiento: {tr.shape}')
    # print(f'Tamaño set de validacion: {vl.shape}')
    # print(f'Tamaño set de prueba: {ts.shape}')

    # Definición de los hiperparámetros INPUT_LENGTH y OUTPUT_LENGTH
    INPUT_LENGTH = 72  # semanas de entrada
    OUTPUT_LENGTH = 1  # semana futura

    # Datasets supervisados para entrenamiento (x_tr, y_tr), validación (x_vl, y_vl) y prueba (x_ts, y_ts)
    x_tr, y_tr = create_supervised_dataset(tr.values, INPUT_LENGTH, OUTPUT_LENGTH, 4)
    x_vl, y_vl = create_supervised_dataset(vl.values, INPUT_LENGTH, OUTPUT_LENGTH, 4)
    x_ts, y_ts = create_supervised_dataset(ts.values, INPUT_LENGTH, OUTPUT_LENGTH, 4)

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

    #y_true = normalization_layer.inverse(y_true)
    #y_true = np.round(y_true)


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

    plt.title("Forecasting", weight='bold', size="16")

    plt.ylabel("T2M", weight='bold', size="14")
    plt.xlabel("Hour", weight='bold', size="14")
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

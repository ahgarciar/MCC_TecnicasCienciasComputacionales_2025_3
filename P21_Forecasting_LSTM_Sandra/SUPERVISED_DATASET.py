import keras.utils
import random as rnd
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from matplotlib import pyplot as plt

from P06_Codificacion.MainCodificacionOneHot import nombre_columnas


def train_val_test_split(serie, tr_size=0.6, vl_size=0.2, ts_size=0.2):
    N = serie.shape[0]
    Ntrain = int(tr_size * N)
    Nval = int(vl_size * N)
    # Ntest => N - Ntrain - Nval

    train = serie[0:Ntrain]
    val = serie[Ntrain:Ntrain + Nval]
    test = serie[Ntrain + Nval:]

    del N
    del Ntrain

    return train, val, test

def create_supervised_dataset(array, input_length, output_length, step=1):
    '''Permite crear un dataset con las entradas (array_x) y sdalidas (arrays_y)
        requeridas por la red LSTM

        Parametros:
        -array: Arreglo numpy de Tamaño N x features (N:Cantidad de datos,
        f: cantidad de features)
        -input_lenth: instantes de tiempo consecutivos de la(s) serie(s) de tiempo
        usados para alimentar al modeo
        -output_length: instantes de tiempo a pronosticar (salida del modelo)
    '''

    array_x, arrays_y = [], []
    shape = array.shape
    if len(shape) == 1:  # si tenemos una sola caracteristica en la serie (univariada)
        rows, cols = array.shape[0], 1
        array = array.reshape(rows, cols)
    else:  # multivariado
        rows, cols = array.shape

    #for i in range(rows - input_length - output_length):
    # 'step' para saltarse ventanas (reduce RAM linealmente con step)
    for i in range(0, rows - input_length - output_length, step):
        array_x.append(array[i:i + input_length, 0:cols])
        arrays_y.append(array[i + input_length:i + input_length + output_length, -1].reshape(output_length, 1))

    array_x = np.array(array_x)
    arrays_y = np.array(arrays_y)

    del shape
    del rows
    del cols
    del array

    return array_x, arrays_y

def load_instance(instance_name=''):
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

        return instance


if __name__ == '__main__':

    rnd.seed(5)
    np.random.seed(5)
    tf.random.set_seed(5)
    keras.utils.set_random_seed(5)

    tf.keras.backend.clear_session()

    # load instance
    instance = load_instance("instancia_sandra.csv")

    tr, vl, ts = train_val_test_split(instance)
    # print(f'Tamaño set de entrenamiento: {tr.shape}')
    # print(f'Tamaño set de validacion: {vl.shape}')
    # print(f'Tamaño set de prueba: {ts.shape}')

    nombre_columnas = instance.columns
    for i in range(len(nombre_columnas)):
        print("Indice: ", i , " Columna: ", nombre_columnas[i])

    indice = input("indic de la columna a graficar: ")
    indice = int(indice)

    nombre_columnas = instance.columns[indice]  # INDICE DE LA COLUMNA A GRAFICAR

    fig, ax = plt.subplots(figsize=(16,5))
    ax.plot(tr[nombre_columnas], label="Train")
    ax.plot(vl[nombre_columnas], label="Validation")
    ax.plot(ts[nombre_columnas], label="Test")
    plt.legend()
    plt.show()


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

import keras.utils
from Parameterset_for_LSTM import ParameterSetLSTM
import os
import random as rnd
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers.legacy import Adam  # para mac m1 y m2
import gc
from tensorflow.keras.saving import register_keras_serializable
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform, Orthogonal, RandomUniform
import dill

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Escalando cada característica por separado
scalers = dict()  # CREAR GLOBAL

@register_keras_serializable(package='Custom', name='CustomNormalization')
class CustomNormalization(tf.keras.layers.Layer):
    def __init__(self, mean, std):
        super(CustomNormalization, self).__init__()
        self.mean = mean
        self.std = std

    def call(self, inputs):  # Normaliza los inputs
        return (inputs - self.mean) / self.std

    def inverse(self, inputs):  # Desnormaliza los inputs
        return inputs * self.std + self.mean


@register_keras_serializable(package='Custom', name='calc_rmse')
def calc_rmse(y_true, y_pred):
    error = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))
    return error


def create_model(parameters, shape):
    kernel_initializer = glorot_uniform(seed=5)
    recurrent_initializer = Orthogonal(gain=1.0, seed=5)
    bias_initializer = 'zeros'

    model = Sequential()
    #capa 1
    model.add(GRU(units=parameters[0], return_sequences=True,
                  input_shape=(shape[1], shape[2]),
                  kernel_initializer=kernel_initializer,
                  recurrent_initializer=recurrent_initializer,
                  bias_initializer=bias_initializer))
    model.add(Dropout(parameters[6]))
    # capa 2 (última recurrente, sin return_sequences)
    model.add(GRU(units=parameters[1], return_sequences=False,
                  kernel_initializer=kernel_initializer,
                  recurrent_initializer=recurrent_initializer,
                  bias_initializer=bias_initializer))
    model.add(Dropout(parameters[7]))

    model.add(Dense(1, activation='linear'))

    opt = Adam(learning_rate=parameters[5])
    model.compile(optimizer=opt, loss=calc_rmse)

    return model


def train_and_evaluate(parameters, datasets, std):
    shape = datasets['x_tr'].shape
    model = create_model(parameters, shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    historia = model.fit(datasets['x_tr'], datasets['y_tr'],
                         epochs=parameters[3],
                         batch_size=parameters[4],
                         callbacks=[early_stopping],
                         validation_data=(datasets['x_vl'], datasets['y_vl']),
                         shuffle=False,
                         verbose=0)  # verbose = 0 para no ver epocas

    error = model.evaluate(x=datasets['x_vl'], y=datasets['y_vl'], verbose=0)
    ################################################################
    model.save('Mejor_Modelo/modelo_' + parameters[9] + '.keras')

    with open('Mejor_Modelo/history_' + parameters[9] + '.pkl', 'wb') as file:
        dill.dump(historia, file)

    ################################################################
    del model
    del historia
    del early_stopping

    tf.keras.backend.clear_session()
    gc.collect()  # Forzar la recolección de basura

    return error

def exec_lstm(parameters, datasets, std):
        result = train_and_evaluate(parameters, datasets, std)
        return result

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

        return instance_std, normalization_layer


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


if __name__ == '__main__':

    tf.config.experimental.enable_op_determinism()
    tf.keras.backend.set_floatx('float32')

    rnd.seed(5)
    np.random.seed(5)
    tf.random.set_seed(5)
    keras.utils.set_random_seed(5)

    tf.keras.backend.clear_session()

    # load instances
    instance, estandarizador = load_instance("instancia_sandra.csv")

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

    ####################################################################################
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

    max_batch_size = 200 #int(x_tr.shape[0])  # total de registros posibles
    ParameterSetLSTM.min_max_values[4][1] = max_batch_size  # actualiza el nuevo max batch size

    params = []
    for j in range(0, 5):
        vmin = ParameterSetLSTM.min_max_values[j][0]
        vmax = ParameterSetLSTM.min_max_values[j][1]
        params.append(rnd.randint(vmin, vmax))
    for j in range(5, 9):
        vmin = ParameterSetLSTM.min_max_values[j][0]
        vmax = ParameterSetLSTM.min_max_values[j][1]
        val = rnd.uniform(vmin, vmax)
        val  = round(val, 4)
        params.append(val)

    params.append("solution_" + str(0)) #solo para guardar el nombre de la solucion

    results = exec_lstm(parameters=params, datasets=data, std=estandarizador)

    root_mse = results

    print("ParameterSet: ", params, " RMSE:", root_mse)

#ParameterSet:  [209, 115, 239, 32, 186, 0.0094, 0.422, 0.4767, 0.2087, 'solution_0']  RMSE: 0.8286582827568054


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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam  # para mac m1 y m2
import gc
from tensorflow.keras.saving import register_keras_serializable
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform, Orthogonal, RandomUniform

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
    model.add(LSTM(units=parameters[0], activation='relu', return_sequences=True, input_shape=(shape[1], shape[2]),
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer,
                   bias_initializer=bias_initializer,
                   unit_forget_bias=True))
    model.add(Dropout(parameters[6]))
    #capa 2
    model.add(LSTM(units=parameters[1], activation='relu', return_sequences=True,
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer,
                   bias_initializer=bias_initializer,
                   unit_forget_bias=True))
    model.add(Dropout(parameters[7]))
    #capa 3
    model.add(LSTM(units=parameters[2], activation='relu', return_sequences=False,
                   kernel_initializer=kernel_initializer,
                   recurrent_initializer=recurrent_initializer,
                   bias_initializer=bias_initializer,
                   unit_forget_bias=True))
    model.add(Dropout(parameters[8]))

    model.add(Dense(units=1, activation='linear'))  # relu

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
    if os.path.isfile(os.path.join(base_folder, instance_name)) and instance_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(base_folder, instance_name))
        ###################################################################
        # Convierte la columna 'Fecha' a tipo datetime
        df['date'] = pd.to_datetime(df['Fecha'])
        # Obtener la semana, mes y año de "DATE"
        df['week'] = df['date'].dt.isocalendar().week  # 1 al 52 (53) #semana del año
        df['year'] = df['date'].dt.year
        ###################################################################
        df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
        ###################################################################
        instance = df[['year', 'sin_week', 'cos_week', 'Evento', 'Ventas']]

        instance_std = pd.DataFrame([])

        for index in range(len(instance.columns)):  # -2):
            column = instance.columns[index]
            scaler = StandardScaler()
            instance_std[column] = pd.DataFrame(scaler.fit_transform(instance[[column]]), columns=[column])
            scalers.update({column: scaler})

        mean = scalers['Ventas'].mean_
        std = scalers['Ventas'].scale_

        normalization_layer = CustomNormalization(mean, std)

        del df
        del instance
        del base_folder
        del column
        del index
        del instance_name

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


def create_supervised_dataset(array, input_length, output_length):
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

    for i in range(rows - input_length - output_length):
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

    rnd.seed(5)
    np.random.seed(5)
    tf.random.set_seed(5)
    keras.utils.set_random_seed(5)

    tf.keras.backend.clear_session()

    # load instances
    instance, estandarizador = load_instance("Instancia_Producto1.csv")

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

    ##GENERA POBLACION INICIAL DEL GENETICO
    max_batch_size = int(x_tr.shape[0])  # total de registros posibles
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


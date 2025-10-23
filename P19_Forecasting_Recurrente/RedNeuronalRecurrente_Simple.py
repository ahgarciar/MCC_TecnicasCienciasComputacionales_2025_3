import multiprocessing
import keras.utils
from Parameterset import ParameterSetLSTM
import os
import shutil
import random as rnd
import math as m
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam  # para mac m1 y m2
import gc
import dill
from tensorflow.keras.saving import register_keras_serializable
from keras.callbacks import EarlyStopping

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
    model = Sequential()
    # Primera capa recurrente con Dropout
    model.add(SimpleRNN(parameters[0], return_sequences=True, input_shape=(shape[1], shape[2])))
    model.add(Dropout(parameters[6]))

    # Segunda capa recurrente con Dropout
    model.add(SimpleRNN(parameters[1], return_sequences=True))
    model.add(Dropout(parameters[7]))

    # Tercera capa recurrente con Dropout
    model.add(SimpleRNN(parameters[2]))
    model.add(Dropout(parameters[8]))

    model.add(Dense(units=1, activation='linear'))  # relu

    opt = Adam(learning_rate=parameters[5])
    model.compile(optimizer=opt, loss=calc_rmse)

    return model


def train_and_evaluate(parameters, datasets, std):
    shape = datasets['x_tr'].shape
    model = create_model(parameters, shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Entrenamiento del modelo
    historia = model.fit(datasets['x_tr'], datasets['y_tr'],
                         epochs=parameters[3],
                         batch_size=parameters[4],
                         callbacks=[early_stopping],  # , reduce_lr],
                         validation_data=(datasets['x_vl'], datasets['y_vl']),
                         shuffle=False,
                         verbose=0)  # verbose = 0 para no ver entrenamiento

    # Mínimo val_loss - rmse
    loss = historia.history['val_loss']
    min_val_loss = min(loss)
    # Encontrar la época en la que ocurrió el mínimo val_loss
    best_epoch = loss.index(min_val_loss) + 1

    error = model.evaluate(x=datasets['x_vl'], y=datasets['y_vl'], verbose=0)

    ################################################################
    model.save('Modelos_Temporales/modelo_' + parameters[9] + '.keras')

    with open('Modelos_Temporales/history_' + parameters[9] + '.pkl', 'wb') as file:
        dill.dump(historia, file)

    ################################################################
    del model
    del historia
    del early_stopping

    tf.keras.backend.clear_session()
    gc.collect()  # Forzar la recolección de basura

    return error, best_epoch, parameters[9]


def exec(solutions_set, datasets, std):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #with multiprocessing.Pool(processes=4) as pool:
        result = pool.starmap(train_and_evaluate,
                               [(solution.parameters, datasets, std) for solution in solutions_set])
        return result

def repair_solution(value, lower_bound, upper_bound):
    result = value
    if value < lower_bound:
        result = lower_bound
    if value > upper_bound:
        result = upper_bound
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

    tot_solutions = 10  # poblacion
    tot_hijos = 10

    tot_parameters = 9  #
    solutions_parameters = []

    probabilidadCruza = 0.9
    EPS = 1.0e-14
    distributionIndex = 20

    probabilidadMutaDiscreto = 0.20
    probabilidadMutaGen = 0.20  # 1.0 / tot_parameters

    it = 0
    tot_generations = 100

    tf.config.experimental.enable_op_determinism()  # LLLLLLLLLL

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
    # índices de los elementos que cumplen la condición
    condicion = (x_tr[:, -1, 3] > 0) # == 1)  #>0 porque la columna esta estandarizada
    indices = np.where(condicion)[0]

    # multiplica los arreglos seleccionados
    for posicion in range(len(indices)-1, -1, -1):
        # recorre de atras para adelante para evitar que los indices se alteren al ingresar a los nuevos elementos
        idx = indices[posicion]
        veces_duplica = (len(condicion)-len(indices))//len(indices)-1  # -1 porque empieza en 0 el ciclo
        for _ in range(veces_duplica):
        #for _ in range(12):
            x_tr = np.insert(x_tr, idx + 1, x_tr[idx], axis=0)
            y_tr = np.insert(y_tr, idx + 1, y_tr[idx], axis=0)
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

    for i in range(tot_solutions):
        p = []
        for j in range(0, 5):
            vmin = ParameterSetLSTM.min_max_values[j][0]
            vmax = ParameterSetLSTM.min_max_values[j][1]
            p.append(rnd.randint(vmin, vmax))
        for j in range(5, 9):
            vmin = ParameterSetLSTM.min_max_values[j][0]
            vmax = ParameterSetLSTM.min_max_values[j][1]
            p.append(rnd.uniform(vmin, vmax))

        p.append("solution_" + str(i+1))

        solutions_parameters.append(ParameterSetLSTM(init_parameters=p))

    results = exec(solutions_set=solutions_parameters, datasets=data, std=estandarizador)

    for i in range(tot_solutions):
        root_mse, best_epoch, _ = results[i]

        solutions_parameters[i].parameters[3] =  best_epoch + 10 # 10 epocas de margen
        solutions_parameters[i].rmse = root_mse

        print("ParameterSet: ", i+1, " ", solutions_parameters[i].parameters," RMSE: ", root_mse, " best epoch:", best_epoch)
    # population created

    print("Inicia Genetico: ", end="\n")
    cont_mejoras = 1  #
    bestRMSE = 9999

    while it < tot_generations:
        # SELECCION: TORNEO BINARIO
        padres = []
        for i in range(tot_hijos):
            i_padre1 = rnd.randint(0, tot_solutions - 1)
            i_padre2 = i_padre1
            while i_padre2 == i_padre1:
                i_padre2 = rnd.randint(0, tot_solutions - 1)
            p1 = solutions_parameters[i_padre1]
            p2 = solutions_parameters[i_padre2]
            # print("p1: ", p1.fo, "   p2:", p2.fo)
            if p1.rmse < p2.rmse:
                padres.append(p1)
            else:
                padres.append(p2)

            del p1
            del p2

        # CRUZA
        hijos = []
        for k in range(0, tot_hijos, 2):
            p1temp = padres[k]
            p2temp = padres[k + 1]
            hijo1 = ParameterSetLSTM(init_parameters=p1temp.parameters.copy(), init_rmse=p1temp.rmse)
            hijo2 = ParameterSetLSTM(init_parameters=p2temp.parameters.copy(), init_rmse=p2temp.rmse)

            if rnd.random() <= probabilidadCruza:
                # cruza para valores discretos
                for i in range(0, 5):
                    if rnd.random() <= 0.5:
                        hijo1.parameters[i] = p1temp.parameters[i]
                        hijo2.parameters[i] = p2temp.parameters[i]
                    else:
                        hijo1.parameters[i] = p2temp.parameters[i]
                        hijo2.parameters[i] = p1temp.parameters[i]

                # cruza para valores continuos
                for i in range(5, 9):
                    valueX1 = p1temp.parameters[i]
                    valueX2 = p2temp.parameters[i]
                    try:
                        if rnd.random() <= 0.5:
                            if abs(valueX1 - valueX2) > EPS:
                                if valueX1 < valueX2:
                                    y1 = valueX1
                                    y2 = valueX2
                                else:
                                    y1 = valueX2
                                    y2 = valueX1

                                lowerBound = ParameterSetLSTM.min_max_values[i][0]  # p1temp.min_values[i]
                                upperBound = ParameterSetLSTM.min_max_values[i][1]  # p1temp.max_values[i]

                                rand = rnd.random()
                                beta = 1.0 + (2.0 * (y1 - lowerBound) / (y2 - y1))
                                alpha = 2.0 - m.pow(beta, -(distributionIndex + 1.0))

                                if rand <= (1.0 / alpha):
                                    betaq = m.pow(rand * alpha, (1.0 / (distributionIndex + 1.0)))
                                else:
                                    betaq = m.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0))

                                c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

                                beta = 1.0 + (2.0 * (upperBound - y2) / (y2 - y1))
                                alpha = 2.0 - m.pow(beta, -(distributionIndex + 1.0))

                                if rand <= (1.0 / alpha):
                                    betaq = m.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)))
                                else:
                                    betaq = m.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0))

                                c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                                c1 = repair_solution(c1, lowerBound, upperBound)
                                c2 = repair_solution(c2, lowerBound, upperBound)

                                if rnd.random() <= 0.5:
                                    hijo1.parameters[i] = c2
                                    hijo2.parameters[i] = c1
                                else:
                                    hijo1.parameters[i] = c1
                                    hijo2.parameters[i] = c2
                            else:
                                hijo1.parameters[i] = valueX1
                                hijo2.parameters[i] = valueX2
                        else:
                            hijo1.parameters[i] = valueX2
                            hijo2.parameters[i] = valueX1
                    except Exception as ex:
                        print("-", end="")

            hijos.append(hijo1)
            hijos.append(hijo2)

            del p1temp
            del p2temp
            del hijo1
            del hijo2

        # MUTACION
        for k in range(tot_hijos):
            # muta para valores discretos
            for i in range(0, 5):
                if rnd.random() <= probabilidadMutaDiscreto:
                    yl = ParameterSetLSTM.min_max_values[i][0]
                    yu = ParameterSetLSTM.min_max_values[i][1]
                    y = rnd.randint(yl, yu)
                    hijos[k].parameters[i] = y
            # muta para valores continuos
            for i in range(5, 9):
                if rnd.random() <= probabilidadMutaGen:
                    y = hijos[k].parameters[i]
                    yl = ParameterSetLSTM.min_max_values[i][0]  # hijos[k].min_values[i]
                    yu = ParameterSetLSTM.min_max_values[i][1]  # hijos[k].max_values[i]
                    if yl == yu:
                        y = yl
                    else:
                        delta1 = (y - yl) / (yu - yl)
                        delta2 = (yu - y) / (yu - yl)
                        rand = rnd.random()
                        mutPow = 1.0 / (distributionIndex + 1.0)

                        if rand <= 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (m.pow(xy, distributionIndex + 1.0))
                            deltaq = m.pow(val, mutPow) - 1.0
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (m.pow(xy, distributionIndex + 1.0))
                            deltaq = 1.0 - m.pow(val, mutPow)

                        y = y + deltaq * (yu - yl)
                        y = repair_solution(y, yl, yu)

                    hijos[k].parameters[i] = y

        # FO hijos
        #Cambia el nombre del padre por el de los hijos... usado para guardar los modelos
        for i in range(tot_hijos):
            hijos[i].parameters[9]= "child_" + str(i + 1)

        results = exec(solutions_set=hijos, datasets=data, std=estandarizador)

        for i in range(tot_hijos):
            root_mse, best_epoch, _ = results[i]

            hijos[i].parameters[3] = best_epoch + 10  # 10 epocas de margen
            hijos[i].rmse = root_mse

            #print("ParameterSet: ", i + 1, " ", p, " fo:", fo, " best epoch:", best_epoch)
            # print(f"Parameters del hijo {i + 1}: {hijos[i].parameters}, fo: {fo}")

        for i in range(tot_hijos):
            solutions_parameters.append(hijos[i])

        # Ordenar Soluciones por FO y Eliminar Peores
        solutions_parameters.sort()  # ordena de menor a mayor

        while len(solutions_parameters) > tot_solutions:  # (int i = 0; i < tot_hijos; i++):
            solutions_parameters.pop()

        if bestRMSE>solutions_parameters[0].rmse:
            bestRMSE = solutions_parameters[0].rmse
            name = solutions_parameters[0].parameters[9]
            # copia al modelo en una ubicacion diferente
            shutil.copy('Modelos_Temporales/modelo_' + name + '.keras', 'Mejor_Modelo/modelo_mejora_' + str(cont_mejoras) + '.keras')
            shutil.copy('Modelos_Temporales/history_' + name + '.pkl', 'Mejor_Modelo/history_mejora_' + str(cont_mejoras) + '.pkl')
            cont_mejoras += 1

            print("Modelo e Historial actualizado con el nuevo best...", end=" ")

        # ELIMINA a la carpeta modelos con su contenido completo y despues recrea la carpeta
        shutil.rmtree("Modelos_Temporales")
        os.makedirs("Modelos_Temporales")

        del padres
        del hijos

        gc.collect()  # Forzar la recolección de basura

        print("It: ", it + 1, " - RMSE: ", bestRMSE, " - Params: ", solutions_parameters[0].parameters)

        it += 1

    print("\nBest Values: ")
    print("RMSE: ", bestRMSE, " - ", solutions_parameters[0].parameters)

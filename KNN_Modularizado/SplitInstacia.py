import pandas as pd

def split_instance(instance, porcentajeEntrenamiento):
    totRegEntrenamiento = int(len(instance) * porcentajeEntrenamiento)
    totRegPrueba = len(instance) - totRegEntrenamiento

    import random
    indices = [i for i in range(len(instance))]
    random.shuffle(indices)

    entrenamiento = pd.DataFrame([])
    for i in range(totRegEntrenamiento):
        registro = instance.loc[[indices[i]]]
        entrenamiento = pd.concat([entrenamiento, registro])
        # print()

    prueba = pd.DataFrame([])
    for i in range(totRegPrueba):
        registro = instance.loc[[indices[i + totRegEntrenamiento]]]
        prueba = pd.concat([prueba, registro])

    # print()
    entrenamiento.reset_index(inplace=True, drop=True)
    prueba.reset_index(inplace=True, drop=True)

    return entrenamiento, prueba
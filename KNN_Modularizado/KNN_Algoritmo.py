import MetricasSimilitud

def exec(instancia, registro_comp, k):
    # para crear columna distancia  con el valor por defecto de 0
    instancia["distancia"] = 0  # instancia["distancia"]

    #print(" resultado de la comparación:..")
    for i in range(0, len(instancia)):
        registro = list(instancia.loc[i].to_dict().values())
        dist = MetricasSimilitud.getDistancia(registro_comp, registro, tipo=0)

        instancia.loc[i, "distancia"] = dist  # se actualiza el valor de la columna para el registro
        # con base en la distancia calculada

     #   print(str(registro) + " Distancia: " + str(dist))

    # ordena la instancia de menor a mayor distancia
    instancia = instancia.sort_values(by="distancia", ascending=True)

    instancia = instancia.reset_index()

    clasesK = []
    for i in range(0, k):  # Desde el 1, porque el valor 0 es el comparacion
        clase = instancia.loc[i]["Play Tennis"]
        clasesK.append(clase)
    #print("Clases K: " + str(clasesK))

    import statistics
    moda = statistics.mode(clasesK)
    #print(moda)

    return moda

def probarKnn(entrenamiento, prueba, k):
    respuestas =[]
    for i in range(len(prueba)):
        registro = list(prueba.loc[i])
        respuesta = exec(entrenamiento, registro, k)
        respuestas.append(respuesta)
    return respuestas
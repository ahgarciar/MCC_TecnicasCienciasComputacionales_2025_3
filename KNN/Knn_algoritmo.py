def hamming(vector1, vector2):
    distancia = 0
    for i in range(1, len(vector1)-1):
        if vector1[i] != vector2[i]:
            distancia+=1
    return distancia

def exec(instancia, k):
    # print(instancia)
    # >>>>>>para probar... se utilizara el primer registro de la instancia<<<<
    # comparacion del registro 1 con todos los demas registros
    print("Registro a comparar:")
    registro_comp = list(instancia.loc[0].to_dict().values())
    print(registro_comp)

    # para crear columna distancia  con el valor por defecto de 0
    instancia["distancia"] = 0  # instancia["distancia"]

    print(" resultado de la comparación:..")
    for i in range(1, len(instancia)):
        registro = list(instancia.loc[i].to_dict().values())
        dist = hamming(registro_comp, registro)

        instancia.loc[i, "distancia"] = dist  # se actualiza el valor de la columna para el registro
        # con base en la distancia calculada

        print(str(registro) + " Distancia: " + str(dist))

    # ordena la instancia de menor a mayor distancia
    instancia = instancia.sort_values(by="distancia", ascending=True)

    instancia = instancia.reset_index()

    clasesK = []
    for i in range(1, k + 1):  # Desde el 1, porque el valor 0 es el comparacion
        clase = instancia.loc[i]["Play Tennis"]
        clasesK.append(clase)
    print("Clases K: " + str(clasesK))

    import statistics
    moda = statistics.mode(clasesK)
    print(moda)

    return moda
from P05_KNN_Modularizado import CargaInstancia
instancia = CargaInstancia.cargarInstancia("../Archivos/iris/iris.csv")
#############################################################################################
###GRUPOS A GENERAR
v_K = 6    # Pagina de referencia comparativa ->>>>  https://orange.readthedocs.io/en/latest/reference/rst/Orange.feature.discretization.html#Orange.feature.discretization.Discretization
#############################################################################################
columns = instancia.columns[:-1] #nombre de atributos sin la clase

for column in columns: #por columna o atributo
    aux = 0
    #ordenar los elementos
    instancia.sort_values(column, inplace = True)
    instancia.reset_index(inplace=True, drop=True)
    ######################################################################
    instancia[column] = instancia[column].astype("str")
    for k in range(v_K):
        for j in range(int(len(instancia)/v_K)):
            instancia.loc[aux, column] = "var" + str(k+1)
            aux += 1
         #
print("")

instancia.to_csv("../Archivos/iris/instancia_discretizada_EFB.csv", index=False)
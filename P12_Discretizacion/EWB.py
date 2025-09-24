from sympy.integrals.manualintegrate import constant_rule

from P05_KNN_Modularizado import CargaInstancia

instancia = CargaInstancia.cargarInstancia("../Archivos/iris/iris.csv")

X = instancia.iloc[:,:-1]
y = instancia.iloc[:,-1]
#############################################################################################
###GRUPOS A GENERAR
v_K = 6    # Pagina de referencia comparativa ->>>>  https://orange.readthedocs.io/en/latest/reference/rst/Orange.feature.discretization.html#Orange.feature.discretization.Discretization
#############################################################################################
intervalos = []
columns = X.columns
for column in columns: #por columna o atributo
    auxiliar = X[column]
    v_max = max(auxiliar)
    v_min = min(auxiliar)
    v_width = round((v_max-v_min)/v_K,4)
    ######################################################################
    print("Atributo analizado:" , column)
    print("min: ", v_min)
    print("max: ", v_max)    
    print("width: ", v_width)
    ######################################################################    
    control  = round(v_min+v_width,4)
    #max extremo izquierdo
    temp = [{"inferior": 0, "superior": control}]
    for j in range(1,v_K-1):
        control2 = round(control + v_width, 4)
        s = {"inferior": control, "superior": control2}
        control = control2
        temp.append(s)
    #max extremo derecho
    ultimo = {"inferior": control, "superior": 999}
    temp.append(ultimo)
    intervalos.append(temp)

    X[column] = X[column].astype("str")
    for j in range(len(X)):
        val = float(X.iloc[j][column])
        for k in range(len(temp)):
            inferior = temp[k]["inferior"]
            superior = temp[k]["superior"]
            if val >= inferior and val<superior:
                X.loc[j, column] = "var" + str(k+1)
         #
    #print()
#############################################################################################
for i in intervalos:
    print(i)

X["class"] = y

X.to_csv("../Archivos/iris/instancia_discretizada_EWB.csv", index=False)
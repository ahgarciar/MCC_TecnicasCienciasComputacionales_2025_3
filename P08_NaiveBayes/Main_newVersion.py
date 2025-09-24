from sklearn.model_selection import train_test_split
from P05_KNN_Modularizado import CargaInstancia

dataset = CargaInstancia.cargarInstancia("../Archivos/iris/instancia_discretizada_EWB.csv")

X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y #APLICA ESTRATIFICACION
)
#############################################################################################
##count registers per class
#############################################################################################
probabilities = []
auxiliar = {}
for register_index in range(len(y_train)):
    label = y_train.iloc[register_index]
    if label in auxiliar:
        auxiliar[label] += 1
    else:
        auxiliar[label] = 1
probabilities.append(auxiliar)
#############################################################################################
##count registers per attribute
#############################################################################################
columns = X_train.columns[:-1]
for column in columns:
    auxiliar = {}
    for register_index in range(len(X_train)):
        v_label = X_train.iloc[register_index][column]
        v_class = y_train.iloc[register_index]
        if (v_label,v_class) in auxiliar:
            auxiliar[(v_label,v_class)] += 1
        else:
            auxiliar[(v_label,v_class)] = 1
            #print(v_label, "  ", v_class)
    probabilities.append(auxiliar)
#############################################################################################
##calculate probabilities per attribute
#############################################################################################
for index in range(1, len(probabilities)):
    for c in probabilities[index]: #per attribute
        #print(probabilities[0][c[1]])
        probabilities[index][c] =  probabilities[index][c]/probabilities[0][c[1]]
    #print(probabilities[0][c])
#############################################################################################
##calculate probabilities per class
#############################################################################################
for c in probabilities[0]:
    probabilities[0][c] = probabilities[0][c]/len(X_train)
    #print(probabilities[0][c])
#############################################################################################
#############################################################################################
## TESTING
#############################################################################################
correct_classify = 0

for k in range(len(X_test)):
    register = X_test.iloc[k]
    sum = 0
    probabilities_per_class = {}
    for c in probabilities[0]:
        #print(c)
        auxiliar = probabilities[0][c]
        for index in range(1, len(probabilities)):
            col = columns[index-1]
            if  (register[col], c) in probabilities[index]:
                auxiliar *= probabilities[index][(register[col], c)]
            else:
                auxiliar = 0 #nullify the product
        sum += auxiliar
        probabilities_per_class[c] = auxiliar 

    max = -9999
    c_toAssign = ""
    for p in probabilities_per_class:
        probabilities_per_class[p] = probabilities_per_class[p]/sum
        if probabilities_per_class[p] > max:
            max = probabilities_per_class[p]
            c_toAssign = p
    #############################################################################################
    real_class = y_test.iloc[k]
    print("Real Class: ", real_class ,"Assigned Class: ", c_toAssign, " Probability: ", round(max*100,4), "%")

    if real_class == c_toAssign:
        correct_classify +=1

print("\n\nCorrect Classify: " , correct_classify, " Total Evaluated: " , len(X_test), " Efficiency: ", round(correct_classify/len(X_test)*100,4),"%")
    

def exec(respReales, respAlgoritmo):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(respReales)):
        if respReales[i] == "Yes":  ##REAL
            if respAlgoritmo[i] == "Yes":  # KNN
                TP += 1
            else:  # No
                FN += 1
        else:  # No  #REAL
            if respAlgoritmo[i] == "Yes":  ## KN
                FP += 1
            else:  # No
                TN += 1

    eficiencia = (TP + TN) / (TP + FN + FP + TN)
    #precision = TP / (TP + FP)
    #recall = TP / (TP + FN)
    #f1_score = 2 * ((precision * recall) / (precision + recall))

    #por fines de prueba se comentaron las ecuaciones....
    precision = 0
    recall = 0
    f1_score = 0

    #print("Tot Correctas: " + str(TP + TN))
    #print("Tot Pruebas: " + str(len(respReales)))

    #print("Eficiencia: " + str(eficiencia))
    #print("Precision: " + str(precision))
    #print("Recall: " + str(recall))
    #print("F1-Score: " + str(f1_score))

    return (eficiencia, precision, recall, f1_score)
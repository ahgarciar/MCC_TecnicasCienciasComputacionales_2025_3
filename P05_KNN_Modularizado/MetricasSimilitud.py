
import math as m

#por defecto calcula la distancia manhattan
def getDistancia(vectorA, vectorB, tipo = 1, p = 1, vectorW = []):

        match tipo:
            case 0:
                distancia = __hamming(vectorA, vectorB)
            case 1:
                distancia = __Manhattan(vectorA, vectorB)
            case 2:
                distancia = __Euclideana(vectorA, vectorB)
            case 3:
                distancia = __Chebycheff(vectorA, vectorB)
            case 4:
                distancia = __Coseno(vectorA, vectorB)
            case 5:
                distancia = __EuclideanaPromedio(vectorA, vectorB)
            case 6:
                distancia = __Orloci(vectorA, vectorB)
            case 7:
                distancia = __DiferenciaDeCaracterPromedio(vectorA, vectorB)
            case 8:
                distancia = __Canberra(vectorA, vectorB)
            case 9:
                distancia = __Sorensen_BrayCurtis(vectorA, vectorB)
            case 10:
                distancia = __CoeficienteCorrelacionPearson(vectorA, vectorB)
            case 11:
                distancia = __Minkowski(vectorA, vectorB, p)
            case 12:
                distancia = __EuclideanaPesada(vectorA, vectorB, vectorW)
            case _:
                distancia = -1;

        return distancia

def __hamming(vectorA, vectorB):
    distancia = 0
    for i in range(0, len(vectorA)-1): #para las otras metricas considerar el -1 para no agregar la clase
        if vectorA[i] != vectorB[i]:
            distancia+=1
    return distancia

# Norma 1
def __Manhattan(vectorA, vectorB):
    distancia = 0

    for i in range(len(vectorA)-1):
        distancia += abs(vectorA[i] - vectorB[i])

    return distancia


# Norma 2
def __Euclideana(vectorA, vectorB):
    distancia = 0

    for i in range(len(vectorA)-1):
        distancia += m.pow(vectorA[i] - vectorB[i], 2)

    distancia = m.sqrt(distancia)

    return distancia

#Norma Inf
def __Chebycheff(vectorA, vectorB):
    distancia = 0
    for i in range(len(vectorA)-1):
        aux = abs(vectorA[i] - vectorB[i]);
        if aux > distancia:
            distancia = aux

    return distancia


# Norma Lp
def __Minkowski(vectorA, vectorB, p):
    distancia = 0

    for i in range(len(vectorA)):
        distancia += m.pow(abs(vectorA[i] - vectorB[i]), p)

    distancia = m.pow(distancia, 1 / p)

    return distancia


def __EuclideanaPromedio(vectorA, vectorB):
    distancia = 0

    for i in range(len(vectorA)):
        distancia += m.pow(vectorA[i] - vectorB[i], 2)

    distancia = distancia / len(vectorA)
    distancia = m.sqrt(distancia)

    return distancia


def __EuclideanaPesada(vectorA, vectorB, vectorW):
    distancia = 0

    for i in range(len(vectorA)):
        distancia += vectorW[i] * m.pow(vectorA[i] - vectorB[i], 2)

    distancia = m.sqrt(distancia)

    return distancia


def __DiferenciaDeCaracterPromedio(vectorA, vectorB):
    distancia = 0

    for i in range(len(vectorA)):
        distancia += m.abs(vectorA[i] - vectorB[i])

    distancia = distancia / len(vectorA)

    return distancia


def __Canberra(vectorA, vectorB):
    distancia = 0

    for i in range(len(vectorA)):
        distancia += abs(vectorA[i] - vectorB[i]) / (abs(vectorA[i]) + abs(vectorB[i]))

    return distancia


def __Sorensen_BrayCurtis(vectorA, vectorB):
    d1 = 0
    d2 = 0

    for i in range(len(vectorA)):
        d1 += abs(vectorA[i] - vectorB[i])
        d2 += vectorA[i] + vectorB[i]

    distancia = d1 / (2 + d2)

    return distancia


def __Coseno(vectorA, vectorB):
    distancia = 0
    NormaA = 0
    NormaB = 0

    for i in range(len(vectorA)):
        distancia += vectorA[i] * vectorB[i]
        NormaA += m.pow(vectorA[i], 2)
        NormaB += m.pow(vectorB[i], 2)

    distancia = distancia / (NormaA * NormaB)

    return distancia


def __Orloci(vectorA, vectorB):
    distancia = 0
    NormaA = 0
    NormaB = 0

    for i in range(len(vectorA)):
        distancia += vectorA[i] * vectorB[i]
        NormaA += m.pow(vectorA[i], 2)
        NormaB += m.pow(vectorB[i], 2)

    distancia = 2 - 2 * (distancia / (NormaA * NormaB))

    return distancia


def __CoeficienteCorrelacionPearson(vectorA, vectorB):
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0

    for i in range(len(vectorA)):
        d1 += vectorA[i] * vectorB[i]
        d2 += vectorA[i]
        d3 += vectorB[i]

        d4 += m.pow(vectorA[i], 2)
        d5 += m.pow(vectorB[i], 2)

    distancia = (len(vectorA) * d1 - d2 * d3) / (m.sqrt(d4 * len(vectorA) - m.pow(d2, 2)) * m.sqrt(d5 * len(vectorA) - m.pow(d3, 2)))

    return distancia

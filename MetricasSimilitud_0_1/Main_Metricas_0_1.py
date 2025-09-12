
def getDistancia(tipo, A, B):
    distancia = 0

    match (tipo):
        case 1:
            distancia = Jaccard(A, B)

    return distancia

def Jaccard (A, B):
    dist = 0
    return dist


if __name__ == "__main__":
    ex1 = [False ,  True, True]
    ex2 = [True, False , True]

    dist = Jaccard(ex1, ex2)

    print(dist)




import pandas as pd

def cargarInstancia(nombreInstancia=""):
    instancia = None
    if nombreInstancia != "":
        instancia = pd.read_csv(nombreInstancia)
    return instancia


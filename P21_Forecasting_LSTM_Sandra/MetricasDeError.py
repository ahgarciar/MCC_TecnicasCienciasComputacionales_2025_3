
import numpy as np

def calcMAE(valores_reales, valores_estimados): # Error Absoluto Medio (MAE)
    # Mide el error medio en las mismas unidades que los datos reales. Más bajo es mejor
    MAE = np.mean(np.abs(valores_reales - valores_estimados))
    return MAE


def calcMSE(valores_reales, valores_estimados): # Error Cuadrático Medio (MSE)
    # Penaliza más a los errores grandes
    MSE = np.mean((valores_reales - valores_estimados) ** 2)
    return MSE


def calcRMSE(valores_reales, valores_estimados): # Raíz del Error Cuadrático Medio (RMSE)
    # Penaliza más a los errores grandes. Pone al error en las mismas unidades que los datos reales
    MSE = calcMSE(valores_reales, valores_estimados)
    RMSE = np.sqrt(MSE)
    return RMSE


def calcMAPE(valores_reales, valores_estimados): # Error Porcentual Absoluto Medio (MAPE)
    # Mide el error en porcentaje. Facilita la interpretación en términos relativos
    MAPE = np.mean(np.abs((valores_reales - valores_estimados) / valores_reales)) * 100
    return MAPE
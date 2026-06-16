import pytest
import pandas as pd
import numpy as np
import joblib
from app import preprocesar_entrada

# Cargamos los artefactos reales como una estructura fija (fixture) para los tests
@pytest.fixture(scope="module")
def recursos_modelo():
    modelo = joblib.load('modelo_ridge_house_prices.pkl')
    scaler = joblib.load('escalador_house_prices.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, scaler, columnas

# TEST 1: Verificar las dimensiones exactas (Las 258 columnas requeridas por Ridge)
def test_dimensiones_matriz_final(recursos_modelo):
    _, scaler, columnas = recursos_modelo
    inputs = {
        'OverallQual': 6, 'GrLivArea': 1500, 'GarageCars': 2,
        'TotalBsmtSF': 1000, 'YearBuilt': 1990, 'FullBath': 2, 'Fireplaces': 1
    }
    df_resultado = preprocesar_entrada(inputs, columnas, scaler)
    
    assert df_resultado.shape[0] == 1, "Debería generar exactamente 1 fila de scoring."
    assert df_resultado.shape[1] == 258, "Error crítico: El DataFrame no tiene las 258 columnas reglamentarias."

# TEST 2: Comprobar robustez ante datos ausentes (Estrategia de inyección de medias)
def test_resiliencia_datos_incompletos(recursos_modelo):
    _, scaler, columnas = recursos_modelo
    # Simulamos que por un error de red o interfaz faltan parámetros clave
    inputs_corruptos = {
        'GrLivArea': 1200, 'YearBuilt': 1995
    }
    
    # No debería lanzar KeyError; debería rellenar los huecos con scaler.mean_
    try:
        df_resultado = preprocesar_entrada(inputs_corruptos, columnas, scaler)
        ejecucion_limpia = True
    except KeyError:
        ejecucion_limpia = False
        
    assert ejecucion_limpia is True, "El preprocesamiento falló al recibir un diccionario de características incompleto."
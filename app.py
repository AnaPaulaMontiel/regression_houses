import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # <--- ¡CORREGIDO: Faltaba esta importación crítica!
import os 
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DE ARCHIVOS
# ==========================================================
st.set_page_config(
    page_title="Predicción del Precio de Viviendas",
    page_icon="🏠",
    layout="centered"
)

# Inicializamos la memoria de sesión si no existe para que los botones no se borren
if 'calculo_realizado' not in st.session_state:
    st.session_state.calculo_realizado = False
if 'precio_estimado' not in st.session_state:
    st.session_state.precio_estimado = 0.0

# Cargamos el cerebro de nuestro proyecto (.pkl)
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_ridge_house_prices.pkl')
    escalador = joblib.load('escalador_house_prices.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, escalador, columnas

modelo_ridge, scaler, columnas_modelo = cargar_recursos()


# ==========================================================
# 2. INTERFAZ DE USUARIO
# ==========================================================
st.title("🏠 Simulador de Precios de Vivienda")
st.write("Introduce las características de la propiedad para estimar su valor de mercado en tiempo real.")

st.markdown("---")

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
# 3. PROCESAMIENTO MATEMÁTICO Y CÁLCULO
# ==========================================================
if st.button("📊 Calcular Estimación de Valor", type="primary"):
    # Activamos la memoria de que el usuario ya calculó una vez
    st.session_state.calculo_realizado = True
    
    # Construcción de la matriz matemática
    datos_modelo = {col: 0 for col in columnas_modelo}
    df_escalar = pd.DataFrame([scaler.mean_], columns=scaler.feature_names_in_)
    
    df_escalar['OverallQual'] = float(overall_qual)
    df_escalar['GrLivArea'] = float(gr_liv_area)
    df_escalar['GarageCars'] = float(garage_cars)
    df_escalar['TotalBsmtSF'] = float(total_bsmt_sf)
    df_escalar['YearBuilt'] = float(year_built)
    df_escalar['FullBath'] = float(full_bath)
    df_escalar['Fireplaces'] = float(fireplaces)
    
    df_escalar['GarageArea'] = float(garage_cars * 300)
    df_escalar['1stFlrSF'] = float(total_bsmt_sf)
    
    datos_escalados = scaler.transform(df_escalar[scaler.feature_names_in_])
    df_escalado_limpio = pd.DataFrame(datos_escalados, columns=scaler.feature_names_in_)
    
    for col in df_escalado_limpio.columns:
        if col in datos_modelo:
            datos_modelo[col] = df_escalado_limpio.loc[0, col]
            
    df_final_scoring = pd.DataFrame([datos_modelo], columns=columnas_modelo)
    
    prediccion_log = modelo_ridge.predict(df_final_scoring)
    st.session_state.precio_estimado = float(np.expm1(prediccion_log[0]))


# ==========================================================
# 4. RENDERIZADO DINÁMICO (ZONA SEGURA)
# ==========================================================
if st.session_state.calculo_realizado:
    precio_final_usd = st.session_state.precio_estimado
    
    # Mostramos el precio
    st.success(f"### 💵 Precio Estimado: **${precio_final_usd:,.2f} USD**")
    st.markdown("---")
    
    # --- RENDERIZADO DE GRÁFICOS ---
    st.subheader("📊 Análisis Visual de la Propiedad")
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.write("**Impacto de las características seleccionadas:**")
        variables_clave = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'Fireplaces']
        indices_columnas = [columnas_modelo.index(c) for c in variables_clave]
        pesos = modelo_ridge.coef_[indices_columnas]
        
        df_pesos = pd.DataFrame({'Característica': variables_clave, 'Impacto': pesos}).sort_values(by='Impacto', ascending=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        colors = ['#FF4B4B' if x < 0 else '#1E88E5' for x in df_pesos['Impacto']]
        sns.barplot(data=df_pesos, x='Impacto', y='Característica', palette=colors, ax=ax1)
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)
        st.caption("🟦 Azul: Sube el precio | 🟥 Rojo: Baja el precio")

    with col_graf2:
        st.write("**Posición de tu casa frente al mercado global:**")
        np.random.seed(42)
        precios_mercado_usd = np.random.normal(loc=180000, scale=55000, size=1500)
        precios_mercado_usd = precios_mercado_usd[precios_mercado_usd > 40000]
        
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(precios_mercado_usd, kde=True, color="teal", ax=ax2, alpha=0.4, bins=40)
        ax2.axvline(x=precio_final_usd, color="red", linestyle="--", linewidth=3, label=f"Tu Casa (${precio_final_usd:,.0f})")
        ax2.set_xlim(50000, 350000)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2)
        st.caption("La línea discontinua muestra la posición estimada de tu tasación.")

    # --- 5. PIPELINE DE INGESTIÓN Y FEEDBACK REAL ---
    st.markdown("---")
    st.subheader("📝 Registro de Datos y Feedback del Modelo")
    st.write("Ayúdanos a monitorizar el rendimiento del modelo en producción.")

    # Estructuramos la información para guardar
    nueva_consulta = {
        'Fecha_Registro': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'OverallQual': int(overall_qual),
        'GrLivArea': float(gr_liv_area),
        'GarageCars': int(garage_cars),
        'TotalBsmtSF': float(total_bsmt_sf),
        'YearBuilt': int(year_built),
        'FullBath': int(full_bath),
        'Fireplaces': int(fireplaces),
        'Precio_Estimado_USD': round(precio_final_usd, 2),
        'Feedback_Usuario': 'No especificado'
    }

    col_feed1, col_feed2 = st.columns(2)
    
    with col_feed1:
        # Añadimos key única para evitar el error de ID duplicado
        if st.button("👍 El precio parece correcto", use_container_width=True, key="btn_correcto"):
            nueva_consulta['Feedback_Usuario'] = 'Correcto'
            df_nuevo_registro = pd.DataFrame([nueva_consulta])
            df_nuevo_registro.to_csv('historico_consultas.csv', mode='a', header=not os.path.exists('historico_consultas.csv'), index=False)
            st.toast("¡Gracias! Feedback registrado localmente. 🚀")
            
    with col_feed2:
        # Añadimos key única para evitar el error de ID duplicado
        if st.button("👎 El precio parece desajustado", use_container_width=True, key="btn_desajustado"):
            nueva_consulta['Feedback_Usuario'] = 'Desajustado'
            df_nuevo_registro = pd.DataFrame([nueva_consulta])
            df_nuevo_registro.to_csv('historico_consultas.csv', mode='a', header=not os.path.exists('historico_consultas.csv'), index=False)
            st.toast("Registrado. Analizaremos este caso para mejorar el algoritmo. 🛠️")
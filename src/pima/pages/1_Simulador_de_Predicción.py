import streamlit as st
import pandas as pd
import numpy as np

# Importamos las funciones cacheables desde el módulo compartido
from data_model import load_data, train_model 

# Cargar datos y modelo cacheado 
df = load_data()
model, _, _ = train_model(df) 

# --- Configuración de la Página de Predicción ---
st.title("Simulador Interactivo de Predicción de Diabetes 💉")
st.markdown("Utiliza el formulario de abajo para introducir los datos del paciente. La predicción se generará **solo** después de presionar el botón 'Predecir'.")
st.markdown("---")

# Obtener el rango de valores (min, max) del dataset para los sliders
min_max = df.drop(columns=['Outcome']).agg(['min', 'max'])


# 1. Usar st.form para agrupar las entradas y forzar un único envío
with st.form("prediction_form"):
    
    st.subheader("Datos de Entrada del Paciente")

    # Organizar los sliders en tres columnas
    col_input_1, col_input_2, col_input_3 = st.columns(3)
    
    # ------------------ Columna 1 ------------------
    with col_input_1:
        pregnancies = st.slider("Embarazos", 0, 17, int(df['Pregnancies'].mean()), key='p')
        glucose = st.slider("Concentración de Glucosa", int(min_max.loc['min', 'Glucose']), int(min_max.loc['max', 'Glucose']), int(df['Glucose'].mean()), key='g')
        blood_pressure = st.slider("Presión Sanguínea Diastólica", int(min_max.loc['min', 'BloodPressure']), int(min_max.loc['max', 'BloodPressure']), int(df['BloodPressure'].mean()), key='bp')
    
    # ------------------ Columna 2 ------------------
    with col_input_2:
        skin_thickness = st.slider("Grosor de Pliegue Cutáneo (mm)", int(min_max.loc['min', 'SkinThickness']), int(min_max.loc['max', 'SkinThickness']), int(df['SkinThickness'].mean()), key='st')
        insulin = st.slider("Insulina sérica (mu U/ml)", int(min_max.loc['min', 'Insulin']), int(min_max.loc['max', 'Insulin']), int(df['Insulin'].mean()), key='i')
        bmi = st.slider("Índice de Masa Corporal (BMI)", float(min_max.loc['min', 'BMI']), float(min_max.loc['max', 'BMI']), float(df['BMI'].mean()), step=0.1, key='bmi')
    
    # ------------------ Columna 3 ------------------
    with col_input_3:
        diabetes_pedigree = st.slider("Función Pedigree Diabetes", 0.0, 2.5, float(df['DiabetesPedigreeFunction'].mean()), step=0.01, key='dpf')
        age = st.slider("Edad", int(min_max.loc['min', 'Age']), int(min_max.loc['max', 'Age']), int(df['Age'].mean()), key='a')
        st.markdown("---")
        
        # 2. Botón de envío que activa el formulario
        submitted = st.form_submit_button("Predecir Resultado", type="primary", use_container_width=True)

# 3. Mostrar el resultado SÓLO si el formulario ha sido enviado (`submitted` es True)
if submitted:
    
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Realizar la predicción
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader("🎯 Resultado del Modelo de Regresión Logística")
    st.markdown("---")
    
    col_res_1, col_res_2, col_res_3 = st.columns(3)
    
    with col_res_1:
        if prediction == 1:
            st.error(f"**DIABETES** (Outcome = 1)")
        else:
            st.success(f"**NO DIABETES** (Outcome = 0)")

    with col_res_2:
        st.metric(
            label="Probabilidad de Diabetes (P(y=1))", 
            value=f"{prediction_proba:.2%}",
            delta="Umbral de decisión: 50%"
        )
 
    with col_res_3:
        st.info(
            f"""
            **Datos de Prueba:**
            - Glucosa: **{glucose}**
            - BMI: **{bmi:.1f}**
            - Edad: **{age}**
            """
        )

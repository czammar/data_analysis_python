import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- Carga de Datos (Cacheada) ---
@st.cache_data
def load_data():
    """Carga, limpia e imputa el Pima Indian Diabetes Dataset."""
    data_url = "https://raw.githubusercontent.com/czammar/ai_programming_foundations/refs/heads/main/data/pima_diabetes.csv"
    data = pd.read_csv(data_url)

    cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    # 1. Reemplazar 0s (missing values) por NaN
    data[cols_to_clean] = data[cols_to_clean].replace(0, np.nan)

    # 2. Imputar NaN con la media
    for col in cols_to_clean:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)
        
    return data

# --- Entrenamiento del Modelo (Cacheado) ---
@st.cache_resource 
def train_model(data):
    """Entrena el modelo de Regresión Logística y devuelve el modelo y los datos de prueba."""
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Devolvemos el modelo y los datos de prueba (si son necesarios)
    return model, X_test, y_test
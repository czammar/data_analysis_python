import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importaciones necesarias para métricas y visualización
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Importamos las funciones de carga/entrenamiento desde el módulo compartido
from data_model import load_data, train_model 

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Diabetes Pima Indian",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar datos y entrenar el modelo (cacheados)
df = load_data()
model, X_test, y_test = train_model(df) 

st.title("Aplicación de Análisis Predictivo de Diabetes (Página Principal)")
st.markdown("### Modelo de Clasificación con Regresión Logística")


# --- Sección 1: Análisis y Visualización de Insights ---
def show_eda_insights(data):
    st.header("1. Exploración de Insights del Dataset Pima Indian Diabetes")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Estadísticas",
            "Distribuciones",
            "Análisis Multivariado",
            "Correlación"
        ]
    )

    with tab1:
        st.subheader("Estadísticas Descriptivas")
        st.write(data.describe().T.style.background_gradient(cmap="Blues"))

    with tab2:
        st.subheader("Visualización de Distribuciones por Outcome")

        # Histograma para Glucose
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=data,
            x="Glucose",
            hue="Outcome",
            kde=True,
            bins=25,
            palette={0: "#3498db", 1: "#e74c3c"},
            ax=ax,
        )
        ax.set_title(
            "Distribución de Glucose por Outcome (0: No Diabetes, 1: Diabetes)"
        )
        st.pyplot(fig)
        plt.close(fig)

        # Histograma para BMI
        fig_bmi, ax_bmi = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=data,
            x="BMI",
            hue="Outcome",
            kde=True,
            bins=25,
            palette={0: "#3498db", 1: "#e74c3c"},
            ax=ax_bmi,
        )
        ax_bmi.set_title("Distribución de BMI por Outcome")
        st.pyplot(fig_bmi)
        plt.close(fig_bmi)

    with tab3:
        st.subheader("Relación entre variables")
        pair_plot = sns.pairplot(
            data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]],
            height=2.0,
            diag_kind='kde'
        )
        pair_plot.fig.suptitle("Relación entre variables (Pair Plot)", y=1.02)
        st.pyplot(pair_plot)
        plt.close(pair_plot.fig)

    with tab4:
        st.subheader("Mapa de Calor de Correlación")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Mapa de Calor de Correlación de Características")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            """
            **Relación con la variable `Outcome`:** **Glucose** (0.49) y **IMC** (0.31)
            muestran la correlación más fuerte con la probabilidad de diabetes.
            """
        )


# --- Sección 2: Entrenamiento y Outcomes del Modelo ---
def show_model_results(model, X_test, y_test):
    st.header("2. Outcomes del Modelo de Regresión Logística")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tab_metrics, tab_prob, tab_table = st.tabs(
        ["Métricas y Errores", "Curva ROC y Probabilidades", "Datos y Predicciones"]
    )

    with tab_metrics:
        st.subheader("Matriz de Confusión y Reporte de Clasificación")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["No Diabetes (0)", "Diabetes (1)"],
                yticklabels=["No Diabetes (0)", "Diabetes (1)"],
                ax=ax_cm,
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col2:
            st.markdown("##### Reporte de Clasificación")
            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                target_names=["No Diabetes (0)", "Diabetes (1)"],
            )
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

    with tab_prob:
        st.subheader("Distribución de Predicciones y Curva ROC")

        col_prob1, col_prob2 = st.columns(2)

        with col_prob1:
            st.markdown("##### Distribución de Probabilidades Predichas")
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
            sns.histplot(y_proba[y_test == 0], color="#3498db", kde=True, label="No Diabetes (0)", bins=20, ax=ax_dist)
            sns.histplot(y_proba[y_test == 1], color="#e74c3c", kde=True, label="Diabetes (1)", bins=20, ax=ax_dist)
            ax_dist.set_title("Distribución de Probabilidades Predichas")
            ax_dist.legend()
            st.pyplot(fig_dist)
            plt.close(fig_dist)

        with col_prob2:
            st.markdown("##### Curva ROC (Receiver Operating Characteristic)")
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (área = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Azar")
            ax_roc.set_title("Curva ROC para Regresión Logística")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)

    with tab_table:
        st.subheader("Datos de Prueba y Predicciones del Modelo")
        results_df = X_test.copy()
        results_df["real_Outcome"] = y_test
        results_df["predicted"] = y_pred
        results_df["probability (y=1)"] = y_proba.round(4)
        results_df["Error"] = np.where(
            results_df["real_Outcome"] == results_df["predicted"],
            "Correcto",
            "Incorrecto",
        )
        st.dataframe(
            results_df.sort_values(by="probability (y=1)", ascending=False).head(20),
            use_container_width=True,
        )


# --- Ejecutar las funciones ---
show_eda_insights(df)
st.markdown("---")
show_model_results(model, X_test, y_test)

st.markdown(
    """
    ---
    **Nota:** Usa la página **'Simulador de Predicción'** en el menú de la izquierda
    para probar el modelo con valores personalizados, **sin recargas automáticas**.
    """
)
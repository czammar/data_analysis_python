import streamlit as st
import polars as pl
import requests
from io import StringIO
import plotly.express as px
import pandas as pd
import math # Necesario para el cálculo de promedios

st.set_page_config(page_title="Rutas Aéreas RITA", layout="wide")
st.title("✈️ Análisis de Rutas Aéreas – RITA + OpenFlights")
st.markdown("Carga tu archivo CSV RITA aquí para habilitar el análisis y las métricas.")

# --- Funciones de Carga y Procesamiento ---

@st.cache_data(show_spinner=True)
def load_openflights_airports():
    """Carga y limpia el dataset de aeropuertos de OpenFlights."""
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    colnames = [
        "AirportID", "Name", "City", "Country",
        "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST", "TzDatabaseTimeZone",
        "Type", "Source"
    ]
    
    response = requests.get(url)
    data = StringIO(response.text)
    
    df = pl.read_csv(
        data,
        has_header=False,
        null_values="\\N",
        ignore_errors=True,
        infer_schema_length=20000,
        separator=','
    )
    
    df = df.rename({df.columns[i]: colnames[i] for i in range(len(colnames))})
    
    df = df.select([
        pl.col("IATA").cast(pl.Utf8),
        pl.col("Latitude").cast(pl.Float64),
        pl.col("Longitude").cast(pl.Float64)
    ]).filter(pl.col("IATA").is_not_null())
    
    return df

@st.cache_data(show_spinner=True)
def process_rita_data(uploaded_file, airports_df):
    """Carga el archivo RITA, lo une con aeropuertos y prepara para análisis."""
    df = pl.read_csv(uploaded_file, infer_schema_length=50000)

    # Optimización: Reducir tamaño del DataFrame de Polars antes de uniones
    rename_map = {
        "Origin": "IATA_ORIGIN",
        "Dest": "IATA_DEST",
        "Reporting_Airline": "AIRLINE",
        "FlightDate": "FlightDate" 
    }
    
    cols_to_keep = list(set(rename_map.keys()) | set(["FlightDate"])) 
    df = df.select(cols_to_keep)
    df = df.rename(rename_map)

    # --- CORRECCIÓN DE FECHAS (Dos pasos) ---
    df = df.with_columns(
        pl.col("FlightDate").str.strptime(pl.Date, strict=False).alias("FlightDate")
    )

    df = df.with_columns([
        pl.col("FlightDate").dt.year().alias("Year"),
        pl.col("FlightDate").dt.month().alias("Month")
    ])
    # ----------------------------------------

    # Unir con datos de aeropuertos
    df = df.join(
        airports_df.rename({"IATA": "IATA_ORIGIN", "Latitude": "OriginLat", "Longitude": "OriginLon"}),
        on="IATA_ORIGIN",
        how="left"
    )

    df = df.join(
        airports_df.rename({"IATA": "IATA_DEST", "Latitude": "DestLat", "Longitude": "DestLon"}),
        on="IATA_DEST",
        how="left"
    )

    # Filtrar nulos esenciales
    df = df.drop_nulls(["OriginLat", "OriginLon", "DestLat", "DestLon", "FlightDate"])
    
    return df

# --- Ejecución Principal ---

airports_df = load_openflights_airports()

uploaded_file = st.file_uploader("Sube el archivo CSV RITA", type=["csv"])

if uploaded_file:
    # Procesar el archivo subido y almacenar el DataFrame enriquecido
    with st.spinner("Procesando y enriqueciendo datos de vuelos..."):
        full_df = process_rita_data(uploaded_file, airports_df)
    
    if full_df.is_empty():
        st.error("No se encontraron vuelos válidos después de la limpieza y el enriquecimiento de datos.")
        if 'flight_data' in st.session_state:
            del st.session_state['flight_data']
        st.stop()

    st.session_state['flight_data'] = full_df
    
    st.success("✅ Datos cargados y listos. Ahora puedes usar el menú lateral (si está presente) o continuar con el análisis.")
    
    st.write("### Resumen de Vuelos Cargados")
    st.dataframe(
        full_df.select([
            "FlightDate", "Year", "Month", 
            "IATA_ORIGIN", "IATA_DEST", "AIRLINE",
            "OriginLat", "DestLat"
        ]).head().to_pandas()
    )
    
    # ===================================================
    #   CÁLCULO Y DISPLAY DE MÉTRICAS (REEMPLAZANDO EL GRÁFICO)
    # ===================================================
    st.subheader("📊 Métricas Globales del Dataset Cargado")
    
    # 1. Conteo total de vuelos
    total_vuelos = len(full_df)
    
    # 2. Conteo de días únicos (para el promedio diario)
    num_dias = full_df["FlightDate"].n_unique()
    
    # 3. Promedio de vuelos por día
    promedio_diario = total_vuelos / num_dias if num_dias > 0 else 0
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    col_m1.metric(
        "✈️ Vuelos Totales", 
        f"{total_vuelos:,}",
        help="Número total de vuelos en todo el dataset cargado."
    )
    
    col_m2.metric(
        "📅 Promedio Diario", 
        f"{promedio_diario:.2f}",
        help="Promedio de vuelos por día de operación."
    )
    
    col_m3.metric(
        "🗓️ Días de Operación",
        f"{num_dias:,}",
        help="Días únicos de operación registrados."
    )

else:
    if 'flight_data' in st.session_state:
        del st.session_state['flight_data']
    st.info("Sube el CSV para habilitar el análisis de rutas.")
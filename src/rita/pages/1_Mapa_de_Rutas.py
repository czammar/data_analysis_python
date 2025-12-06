import streamlit as st
import polars as pl
import folium
from folium import PolyLine
from streamlit_folium import st_folium
import plotly.express as px

if 'flight_data' not in st.session_state:
    st.warning("Por favor, carga primero el archivo CSV RITA en la página principal.")
    st.stop()

# Recuperar el DataFrame de Polars desde la caché de sesión
df = st.session_state['flight_data']

st.title("🗺️ Mapa y Análisis de Rutas Filtradas")
st.markdown("Filtra los datos para ver la distribución de vuelos y el mapa interactivo.")

# --- Controles de Filtro (Mes/Año) ---
años = sorted(df["Year"].unique().to_list())
meses = sorted(df["Month"].unique().to_list())

colA, colB = st.columns(2)
# Usamos el estado de sesión para mantener la consistencia entre recargas
if 'selected_year' not in st.session_state:
    st.session_state['selected_year'] = años[0] if años else None
if 'selected_month' not in st.session_state:
    st.session_state['selected_month'] = meses[0] if meses else None

sel_year = colA.selectbox(
    "Selecciona el AÑO", 
    años, 
    key='selected_year'
)
sel_month = colB.selectbox(
    "Selecciona el MES", 
    meses, 
    key='selected_month'
)

# ---- Filtrar por Año/Mes (Base) ----
df_month = df.filter(
    (pl.col("Year") == sel_year) & 
    (pl.col("Month") == sel_month)
)

if df_month.is_empty():
    st.warning(f"⚠️ No hay vuelos para {sel_year}-{sel_month:02d}.")
    st.stop()

# Copia para filtros adicionales
filtered_df = df_month.clone()

# --- Controles de Filtro (Origen/Destino) ---
col1, col2 = st.columns(2)

origenes = sorted(df_month["IATA_ORIGIN"].unique().to_list())
destinos = sorted(df_month["IATA_DEST"].unique().to_list())

sel_origen = col1.selectbox("Filtrar por ORIGEN", ["Todos"] + origenes)
sel_destino = col2.selectbox("Filtrar por DESTINO", ["Todos"] + destinos)

if sel_origen != "Todos":
    filtered_df = filtered_df.filter(pl.col("IATA_ORIGIN") == sel_origen)

if sel_destino != "Todos":
    filtered_df = filtered_df.filter(pl.col("IATA_DEST") == sel_destino)

if filtered_df.is_empty():
    st.warning("No hay vuelos después de aplicar filtros de Origen/Destino.")
    st.stop()

# ===================================================
#   Optimización: Cálculo de Rutas para Mapa
# ===================================================

# Usamos st.cache_data en esta función intensiva de Polars.
# Esto asegura que si solo cambian los selectboxes de Origen/Destino, 
# pero no el Año/Mes, el cálculo no se repite.
@st.cache_data(show_spinner=False)
def calculate_routes_for_map(df_to_analyze):
    """Calcula las rutas únicas con coordenadas y un color único por aerolínea."""
    
    # 1. Agrupar para obtener la cuenta total y el color (optimizado con Polars)
    route_counts = (
        df_to_analyze.group_by([
            "IATA_ORIGIN", "OriginLat", "OriginLon",
            "IATA_DEST", "DestLat", "DestLon",
            "AIRLINE" # Mantener AIRLINE si se necesita el color
        ])
        .count()
        .rename({"count": "total"})
        # 2. Asignar un color único basado en el hash de la aerolínea (si es necesario)
        .with_columns([
            pl.col("AIRLINE").map_elements(
                lambda a: f"#{(abs(hash(a)) & 0xFFFFFF):06x}"
            ).alias("color")
        ])
    )
    
    # 3. Extraer aeropuertos únicos para marcadores
    airports_unique = pl.concat([
        route_counts.select(["IATA_ORIGIN", "OriginLat", "OriginLon"])
                    .rename({"IATA_ORIGIN": "IATA", "OriginLat": "lat", "OriginLon": "lon"}),
        route_counts.select(["IATA_DEST", "DestLat", "DestLon"])
                    .rename({"IATA_DEST": "IATA", "DestLat": "lat", "DestLon": "lon"})
    ]).unique(subset=["IATA"])
    
    # Convertir a Pandas solo para la iteración de Folium
    return route_counts.to_pandas(), airports_unique.to_pandas()

# Ejecutar el cálculo optimizado
route_counts_pd, airports_unique_pd = calculate_routes_for_map(filtered_df)

# ===================================================
#   Visualizaciones y Métricas
# ===================================================
st.subheader("📊 Métricas y Visualizaciones")

total_vuelos = len(filtered_df)

# Top Origen y Destino (uso de .head(1) en Polars es más eficiente que [0, "Col"])
origen_top = (
    filtered_df.group_by("IATA_ORIGIN")
               .count()
               .sort("count", descending=True)
               .head(1).select("IATA_ORIGIN").item()
)

destino_top = (
    filtered_df.group_by("IATA_DEST")
               .count()
               .sort("count", descending=True)
               .head(1).select("IATA_DEST").item()
)

c1, c2, c3 = st.columns(3)
c1.metric("📅 Año–Mes", f"{sel_year}-{sel_month:02d}")
c2.metric("✈️ Total Vuelos Filtrados", total_vuelos)
c3.metric("🛫 Origen más frecuente", origen_top)

c4, c5, c6 = st.columns(3)
c4.metric("🛬 Destino más frecuente", destino_top)

# --- Top 10 Rutas ---
st.subheader("📈 Top 10 Rutas Origen–Destino")

top10 = (
    route_counts_pd.groupby(["IATA_ORIGIN", "IATA_DEST"])["total"].sum()
                   .nlargest(10).reset_index(name="total")
)
top10["Ruta"] = top10["IATA_ORIGIN"] + " → " + top10["IATA_DEST"]

fig_bar = px.bar(
    top10,
    x="Ruta",
    y="total",
    text="total",
    title="Top 10 Rutas del Mes Filtrado"
)
fig_bar.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Distribución de Aerolíneas ---
st.subheader("🧁 Distribución de vuelos por aerolínea")

# Usar la tabla de rutas agregada para obtener los totales de aerolíneas (más eficiente)
airline_counts = (
    route_counts_pd.groupby("AIRLINE")["total"].sum()
                   .reset_index(name="total")
)

fig_pie = px.pie(
    airline_counts,
    names="AIRLINE",
    values="total",
    title="Vuelos por Aerolínea"
)
st.plotly_chart(fig_pie, use_container_width=True)

# --- Serie de Tiempo ---
st.subheader("📅 Serie de tiempo de vuelos diarios")

daily_ts = (
    filtered_df.group_by("FlightDate")
               .count()
               .rename({"count": "vuelos"})
               .sort("FlightDate")
               .to_pandas()
)

fig_ts = px.line(
    daily_ts,
    x="FlightDate",
    y="vuelos",
    markers=True,
    title="Serie diaria del mes filtrado"
)
st.plotly_chart(fig_ts, use_container_width=True)


# ===================================================
#  MAPA FINAL (Separado y optimizado con caché de datos)
# ===================================================

st.subheader(f"🗺️ Rutas del Mes: {sel_year}-{sel_month:02d}")

# 1. Inicializar el mapa
m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB Positron")

# 2. Añadir marcadores de aeropuertos únicos
for _, ap in airports_unique_pd.iterrows():
    folium.CircleMarker(
        location=[ap["lat"], ap["lon"]],
        radius=3, # Aumentar el radio para que sean más visibles
        color="#3498db", # Color azul
        fill=True,
        fill_color="#2980b9",
        fill_opacity=0.8,
        tooltip=ap["IATA"] # Añadir tooltip
    ).add_to(m)

# 3. Añadir líneas de ruta
# Iterar sobre el DataFrame de Pandas de las rutas precalculadas
for row in route_counts_pd.itertuples():
    # El peso (weight) ahora puede ser proporcional al total de vuelos
    # Usar una escala logarítmica o una simple clamp para evitar líneas demasiado gruesas
    weight_scaled = max(0.5, min(5, row.total / route_counts_pd['total'].max() * 4)) 

    PolyLine(
        locations=[[row.OriginLat, row.OriginLon], [row.DestLat, row.DestLon]],
        color=row.color, # Color por aerolínea
        weight=weight_scaled, # Peso dinámico
        opacity=0.6
    ).add_to(m)

# 4. Renderizar el mapa de Folium
# El ancho por defecto de la columna en layout="wide" es 700px.
# Podemos usar 'use_container_width=True' o especificar el tamaño.
st_folium(m, width=1400, height=800)
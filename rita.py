import streamlit as st
import polars as pl
import folium
from folium import PolyLine
import requests
from io import StringIO
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Mapa de Vuelos RITA Animado", layout="wide")
st.title("✈️ Mapa de Rutas – Dataset RITA + OpenFlights")

@st.cache_data(show_spinner=True)
def load_openflights_airports():
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
        infer_schema_length=20000
    )

    df = df.rename({df.columns[i]: colnames[i] for i in range(len(colnames))})

    df = df.select([
        pl.col("IATA").cast(pl.Utf8),
        pl.col("Latitude").cast(pl.Float64),
        pl.col("Longitude").cast(pl.Float64)
    ])

    df = df.filter(pl.col("IATA").is_not_null())

    return df

airports_df = load_openflights_airports()

uploaded_file = st.file_uploader("Sube el archivo CSV RITA", type=["csv"])

if uploaded_file:
    df = pl.read_csv(uploaded_file, infer_schema_length=50000)

    rename_map = {
        "Origin": "IATA_ORIGIN",
        "Dest": "IATA_DEST",
        "Reporting_Airline": "AIRLINE"
    }
    df = df.rename(rename_map)

    df = df.with_columns([
        pl.col("FlightDate").str.strptime(pl.Date, strict=False)
    ])

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

    df = df.drop_nulls(["OriginLat", "OriginLon", "DestLat", "DestLon", "FlightDate"])

    if df.is_empty():
        st.error("No hay vuelos válidos.")
        st.stop()

    st.write("### Datos enriquecidos")
    st.dataframe(df.select([
        "IATA_ORIGIN", "IATA_DEST", "AIRLINE",
        "OriginLat", "OriginLon", "DestLat", "DestLon"
    ]).head().to_pandas())

    # ---- Agregar variables Año / Mes ----
    df = df.with_columns([
        pl.col("FlightDate").dt.year().alias("Year"),
        pl.col("FlightDate").dt.month().alias("Month")
    ])

    años = sorted(df["Year"].unique().to_list())
    meses = sorted(df["Month"].unique().to_list())

    colA, colB = st.columns(2)
    sel_year = colA.selectbox("Selecciona el AÑO", años)
    sel_month = colB.selectbox("Selecciona el MES", meses)

    # ---- Filtrar por mes ----
    df_month = df.filter(
        (pl.col("Year") == sel_year) &
        (pl.col("Month") == sel_month)
    )

    if df_month.is_empty():
        st.warning("⚠️ No hay vuelos para el año/mes seleccionado.")
        st.stop()

    # Copia para filtros adicionales
    filtered_df = df_month.clone()

    # ---- Controles de origen/destino ----
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
        st.warning("No hay vuelos después de aplicar filtros.")
        st.stop()

    # ===================================================
    #   MÉTRICAS BASADAS EN AÑO + MES + FILTROS
    # ===================================================

    total_vuelos = len(filtered_df)

    origen_top = (
        filtered_df.group_by("IATA_ORIGIN")
                   .count()
                   .sort("count", descending=True)[0, "IATA_ORIGIN"]
    )

    destino_top = (
        filtered_df.group_by("IATA_DEST")
                   .count()
                   .sort("count", descending=True)[0, "IATA_DEST"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Año–Mes", f"{sel_year}-{sel_month:02d}")
    c2.metric("✈️ Total vuelos filtrados", total_vuelos)
    c3.metric("🛫 Origen más frecuente", origen_top)

    c4 = st.metric("🛬 Destino más frecuente", destino_top)

    # ===================================================
    #  DISTRIBUCIÓN DE AEROLÍNEAS
    # ===================================================
    st.subheader("🧁 Distribución de vuelos por aerolínea")

    airline_counts = (
        filtered_df.group_by("AIRLINE")
                   .count()
                   .rename({"count": "total"})
                   .to_pandas()
    )

    fig_pie = px.pie(
        airline_counts,
        names="AIRLINE",
        values="total",
        title="Vuelos por aerolínea"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ===================================================
    #  SERIE DE TIEMPO (DIARIA) PARA ESE MES
    # ===================================================
    st.subheader("📈 Serie de tiempo de vuelos diarios")

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
    #  RUTAS (todas dentro del mes filtrado)
    # ===================================================
    route_counts = (
        filtered_df.group_by([
            "IATA_ORIGIN", "OriginLat", "OriginLon",
            "IATA_DEST", "DestLat", "DestLon",
            "AIRLINE"
        ])
        .count()
        .rename({"count": "total"})
        .with_columns([
            pl.col("AIRLINE").map_elements(
                lambda a: f"#{(abs(hash(a)) & 0xFFFFFF):06x}"
            ).alias("color")
        ])
    )

    # ===================================================
    #  TOP 10 RUTAS
    # ===================================================
    st.subheader("📊 Top 10 Rutas Origen–Destino")

    top10 = (
        route_counts.group_by(["IATA_ORIGIN", "IATA_DEST"])
                    .agg(pl.col("total").sum())
                    .sort("total", descending=True)
                    .limit(10)
                    .with_columns([
                        (pl.col("IATA_ORIGIN") + " → " + pl.col("IATA_DEST")).alias("Ruta")
                    ])
                    .to_pandas()
    )

    fig = px.bar(
        top10,
        x="Ruta",
        y="total",
        text="total",
        title="Top 10 Rutas del mes filtrado"
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    #  MAPA FINAL
    # ===================================================
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB Positron")

    airports_unique = pl.concat([
        route_counts.select(["IATA_ORIGIN", "OriginLat", "OriginLon"])
                    .rename({"IATA_ORIGIN": "IATA", "OriginLat": "lat", "OriginLon": "lon"}),
        route_counts.select(["IATA_DEST", "DestLat", "DestLon"])
                    .rename({"IATA_DEST": "IATA", "DestLat": "lat", "DestLon": "lon"})
    ]).unique(subset=["IATA"]).to_pandas()

    for _, ap in airports_unique.iterrows():
        folium.CircleMarker(
            location=[ap["lat"], ap["lon"]],
            radius=1,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=1
        ).add_to(m)

    for row in route_counts.to_pandas().itertuples():
        PolyLine(
            locations=[[row.OriginLat, row.OriginLon], [row.DestLat, row.DestLon]],
            color=row.color,
            weight=row.total*0.1,
            opacity=0.85
        ).add_to(m)

    st.subheader(f"🗺️ Rutas del mes: {sel_year}-{sel_month:02d}")
    st_folium(m, width=1400, height=800)

else:
    st.info("Sube el CSV para generar el mapa.")

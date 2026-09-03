import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import io

st.set_page_config(page_title="Generador IDW Cochinilla", layout="wide", initial_sidebar_state="expanded")

st.title("🐛 Generador de Mapas de Infestación por IDW")
st.markdown("Cargue los datos de muestreo de AppSheet y la capa de lotes para calcular el mapa de calor de cochinilla.")

# --- BARRA LATERAL: CARGA DE ARCHIVOS ---
st.sidebar.header("1. Carga de Datos")

file_csv = st.sidebar.file_uploader("Subir Muestreo AppSheet (CSV)", type=["csv"])
file_geojson = st.sidebar.file_uploader("Subir Lotes Finca (GeoJSON/KML)", type=["geojson", "kml", "json"])

# --- FUNCIÓN INTERPOLACIÓN IDW ---
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((xi, yi)), np.column_stack((x, y)))
    dist = np.where(dist == 0, 1e-10, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    zi = np.dot(weights, z)
    return zi

if file_csv and file_geojson:
    try:
        # CORRECCIÓN DE SEPARADOR CSV (Soporta punto y coma o coma)
        try:
            df_points = pd.read_csv(file_csv, sep=';')
            if len(df_points.columns) <= 1:
                file_csv.seek(0)
                df_points = pd.read_csv(file_csv, sep=',')
        except Exception:
            file_csv.seek(0)
            df_points = pd.read_csv(file_csv, sep=',')

        gdf_lotes = gpd.read_file(file_geojson)

        # Limpieza de nombres de columnas (elimina espacios extras)
        df_points.columns = df_points.columns.str.strip()

        st.sidebar.header("2. Filtros y Parámetros IDW")

        # Filtro por Finca
        col_finca = [c for c in df_points.columns if 'FINCA' in c.upper()]
        if col_finca:
            fincas = df_points[col_finca[0]].dropna().unique()
            finca_sel = st.sidebar.selectbox("Seleccione la Finca", fincas)
            df_finca = df_points[df_points[col_finca[0]] == finca_sel].copy()
        else:
            finca_sel = "General"
            df_finca = df_points.copy()

        # DETECCIÓN DE VARIABLES DE INFESTACIÓN
        col_brotes = [c for c in df_finca.columns if 'BROTE' in c.upper() or 'BROTES' in c.upper()]
        col_macollas = [c for c in df_finca.columns if 'MACOLLA' in c.upper() or 'MACOLLAS' in c.upper()]

        dict_opciones = {}
        if col_brotes:
            dict_opciones["% Brotes Infestados"] = col_brotes[0]
        if col_macollas:
            dict_opciones["% Macollas Infestadas"] = col_macollas[0]

        if not dict_opciones:
            # Fallback en caso de nombres no estándar
            dict_opciones = {c: c for c in df_finca.columns if c not in ['ID', 'NUMERO', 'FECHA', 'I', 'J', 'X', 'Y']}

        var_label = st.sidebar.selectbox("Variable a Interpolar", list(dict_opciones.keys()))
        col_val = dict_opciones[var_label]

        # Parámetros del modelo IDW
        power_idw = st.sidebar.slider("Potencia IDW (p)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        resolution = st.sidebar.slider("Resolución Malla", min_value=50, max_value=250, value=120)

        # Identificar coordenadas Lat/Lon (I/J o X/Y)
        col_lat = 'I' if 'I' in df_finca.columns else ('Y' if 'Y' in df_finca.columns else 'Latitud')
        col_lon = 'J' if 'J' in df_finca.columns else ('X' if 'X' in df_finca.columns else 'Longitud')

        # Limpieza de valores con signo % a float
        if df_finca[col_val].dtype == object:
            df_finca[col_val] = df_finca[col_val].astype(str).str.rstrip('%').astype('float')

        # Filtrar capa de lotes por finca si aplica
        col_finca_geo = [c for c in gdf_lotes.columns if 'FINCA' in c.upper()]
        if col_finca_geo and finca_sel != "General":
            gdf_finca = gdf_lotes[gdf_lotes[col_finca_geo[0]].astype(str).str.lower() == str(finca_sel).lower()]
        else:
            gdf_finca = gdf_lotes

        if gdf_finca.empty:
            gdf_finca = gdf_lotes

        # --- BOTÓN DE GENERACIÓN ---
        if st.sidebar.button("🚀 Generar Mapa IDW", type="primary"):
            x = df_finca[col_lon].values
            y = df_finca[col_lat].values
            z = df_finca[col_val].values

            # Malla de Interpolación
            xmin, ymin, xmax, ymax = gdf_finca.total_bounds
            grid_x, grid_y = np.meshgrid(
                np.linspace(xmin, xmax, resolution),
                np.linspace(ymin, ymax, resolution)
            )

            xi = grid_x.flatten()
            yi = grid_y.flatten()

            # Cálculo IDW
            zi = idw_interpolation(x, y, z, xi, yi, power=power_idw)
            grid_z = zi.reshape(grid_x.shape)

            # RENDERIZADO DEL MAPA
            fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)

            # Capa vectorial lotes
            gdf_finca.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=0.8, zorder=3)

            # Capa IDW
            contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap="YlOrRd", alpha=0.75, zorder=2)
            cbar = plt.colorbar(contour, ax=ax, shrink=0.75)
            cbar.set_label(f"{var_label} (%)", fontsize=10, fontweight='bold')

            # Puntos de Muestreo
            ax.scatter(x, y, c='blue', edgecolors='white', linewidth=0.5, s=30, label='Estaciones de Muestreo', zorder=4)

            # Título
            ax.set_title(f"MAPA DE INTERPOLACIÓN IDW - COCHINILLA\n{var_label.upper()} | FINCA: {str(finca_sel).upper()}", fontsize=12, fontweight='bold', pad=12)
            ax.axis('off')
            ax.legend(loc='lower right')

            # Layout de resultados
            col_map, col_stats = st.columns([3, 1])

            with col_map:
                st.pyplot(fig)

                # Exportación a PDF
                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
                pdf_buffer.seek(0)

                st.download_button(
                    label="📄 Descargar Mapa en PDF",
                    data=pdf_buffer,
                    file_name=f"Mapa_IDW_{var_label.replace(' ', '_')}_{finca_sel}.pdf",
                    mime="application/pdf"
                )

            with col_stats:
                st.subheader("📊 Resumen")
                st.metric("Puntos Evaluados", len(df_finca))
                st.metric("Infestación Máxima", f"{z.max():.1f}%")
                st.metric("Infestación Promedio", f"{z.mean():.1f}%")

    except Exception as e:
        st.error(f"Error al procesar los archivos: {e}")

else:
    st.info("👈 Por favor, sube el archivo CSV de muestreo y el GeoJSON de lotes en la barra lateral para comenzar.")

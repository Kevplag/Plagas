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

# --- BARRA LATERAL: CARGA DE ARCHIVOS Y CONTROLES ---
st.sidebar.header("1. Carga de Datos")

file_csv = st.sidebar.file_uploader("Subir Muestreo AppSheet (CSV)", type=["csv"])
file_geojson = st.sidebar.file_uploader("Subir Lotes Finca (GeoJSON/KML)", type=["geojson", "kml", "json"])

# --- FUNCION INTERPOLACIÓN IDW ---
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((xi, yi)), np.column_stack((x, y)))
    dist = np.where(dist == 0, 1e-10, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    zi = np.dot(weights, z)
    return zi

if file_csv and file_geojson:
    try:
        # Carga de datos
        df_points = pd.read_csv(file_csv)
        gdf_lotes = gpd.read_file(file_geojson)

        st.sidebar.header("2. Filtros y Parámetros IDW")

        # Filtro por Finca
        fincas = df_points['Finca'].unique() if 'Finca' in df_points.columns else ["General"]
        finca_sel = st.sidebar.selectbox("Seleccione la Finca", fincas)

        # Parámetros del modelo
        power_idw = st.sidebar.slider("Potencia IDW (p)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        resolution = st.sidebar.slider("Resolución Malla", min_value=50, max_value=250, value=120)

        # Filtrar DataFrames
        if 'Finca' in df_points.columns:
            df_finca = df_points[df_points['Finca'] == finca_sel]
        else:
            df_finca = df_points

        if 'Finca' in gdf_lotes.columns:
            gdf_finca = gdf_lotes[gdf_lotes['Finca'] == finca_sel]
        else:
            gdf_finca = gdf_lotes

        # Identificar columnas
        col_lat = 'Latitud' if 'Latitud' in df_finca.columns else 'Lat'
        col_lon = 'Longitud' if 'Longitud' in df_finca.columns else 'Lng'
        col_val = 'Infestacion' if 'Infestacion' in df_finca.columns else df_finca.columns[-1]

        # --- BOTÓN DE GENERACIÓN ---
        if st.sidebar.button("🚀 Generar Mapa IDW", type="primary"):
            x = df_finca[col_lon].values
            y = df_finca[col_lat].values
            z = df_finca[col_val].values

            # Crear Malla de Interpolación
            xmin, ymin, xmax, ymax = gdf_finca.total_bounds
            grid_x, grid_y = np.meshgrid(
                np.linspace(xmin, xmax, resolution),
                np.linspace(ymin, ymax, resolution)
            )

            xi = grid_x.flatten()
            yi = grid_y.flatten()

            # Ejecutar IDW
            zi = idw_interpolation(x, y, z, xi, yi, power=power_idw)
            grid_z = zi.reshape(grid_x.shape)

            # --- DIBUJAR MAPA ---
            fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)

            # Capa vectorial lotes
            gdf_finca.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=0.7, zorder=3)

            # Capa IDW
            contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap="YlOrRd", alpha=0.75, zorder=2)
            cbar = plt.colorbar(contour, ax=ax, shrink=0.75)
            cbar.set_label("% Infestación Cochinilla", fontsize=10, fontweight='bold')

            # Puntos de Muestreo
            ax.scatter(x, y, c='black', s=12, label='Puntos Muestreados', zorder=4)

            # Rotulado
            ax.set_title(f"MAPA DE INTERPOLACIÓN IDW - INFESTACIÓN DE COCHINILLA\nFINCA: {str(finca_sel).upper()}", fontsize=12, fontweight='bold', pad=12)
            ax.axis('off')
            ax.legend(loc='lower right')

            # Renderizado
            col_map, col_stats = st.columns([3, 1])

            with col_map:
                st.pyplot(fig)

                # Exportar PDF
                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
                pdf_buffer.seek(0)

                st.download_button(
                    label="📄 Descargar Mapa en PDF",
                    data=pdf_buffer,
                    file_name=f"Mapa_IDW_Cochinilla_{finca_sel}.pdf",
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

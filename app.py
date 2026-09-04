import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.spatial import distance
from shapely.vectorized import contains
from shapely.geometry import shape
import io

st.set_page_config(page_title="Generador IDW Cochinilla", layout="wide", initial_sidebar_state="expanded")

st.title("🐛 Generador de Mapas de Infestación por IDW")

# --- BARRA LATERAL: CARGA DE ARCHIVOS ---
file_csv = st.sidebar.file_uploader("Subir Muestreo AppSheet (CSV)", type=["csv"])
file_geojson = st.sidebar.file_uploader("Subir Lotes Finca (GeoJSON/KML)", type=["geojson", "kml", "json"])

def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((xi, yi)), np.column_stack((x, y)))
    dist = np.where(dist == 0, 1e-10, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    zi = np.dot(weights, z)
    return zi

if file_csv and file_geojson:
    try:
        try:
            df_points = pd.read_csv(file_csv, sep=';')
            if len(df_points.columns) <= 1:
                file_csv.seek(0)
                df_points = pd.read_csv(file_csv, sep=',')
        except Exception:
            file_csv.seek(0)
            df_points = pd.read_csv(file_csv, sep=',')

        gdf_lotes = gpd.read_file(file_geojson)
        if gdf_lotes.crs is not None and gdf_lotes.crs.to_string() != "EPSG:4326":
            gdf_lotes = gdf_lotes.to_crs(epsg=4326)
        elif gdf_lotes.crs is None:
            gdf_lotes.set_crs(epsg=4326, inplace=True)

        df_points.columns = df_points.columns.str.strip()

        col_finca = [c for c in df_points.columns if 'FINCA' in c.upper()]
        finca_sel = df_points[col_finca[0]].dropna().unique()[0] if col_finca else "General"
        df_finca = df_points[df_points[col_finca[0]] == finca_sel].copy() if col_finca else df_points.copy()

        col_brotes = [c for c in df_finca.columns if 'BROTE' in c.upper()]
        col_macollas = [c for c in df_finca.columns if 'MACOLLA' in c.upper()]

        dict_opciones = {}
        if col_brotes: dict_opciones["% Brotes Infestados"] = col_brotes[0]
        if col_macollas: dict_opciones["% Macollas Infestadas"] = col_macollas[0]

        var_label = st.sidebar.selectbox("Variable a Interpolar", list(dict_opciones.keys()))
        col_val = dict_opciones[var_label]

        power_idw = st.sidebar.slider("Potencia IDW (p)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        resolution = st.sidebar.slider("Resolución Malla", min_value=100, max_value=300, value=150)

        for c in df_finca.columns:
            if df_finca[c].dtype == object:
                df_finca[c] = df_finca[c].astype(str).str.replace(',', '.')

        col_lat = 'Y' if 'Y' in df_finca.columns else 'Latitud'
        col_lon = 'X' if 'X' in df_finca.columns else 'Longitud'

        df_finca[col_val] = df_finca[col_val].astype(str).str.rstrip('%')
        df_finca[col_val] = pd.to_numeric(df_finca[col_val], errors='coerce')
        df_finca = df_finca.dropna(subset=[col_lat, col_lon, col_val])

        gdf_finca = gdf_lotes

        if st.sidebar.button("🚀 Generar Mapa IDW", type="primary"):
            x = df_finca[col_lon].to_numpy(dtype=float)
            y = df_finca[col_lat].to_numpy(dtype=float)
            z = df_finca[col_val].to_numpy(dtype=float)

            xmin, ymin, xmax, ymax = gdf_finca.total_bounds
            dx = (xmax - xmin) * 0.05
            dy = (ymax - ymin) * 0.05

            x_range = np.linspace(xmin - dx, xmax + dx, resolution)
            y_range = np.linspace(ymin - dy, ymax + dy, resolution)
            grid_x, grid_y = np.meshgrid(x_range, y_range)

            xi = grid_x.flatten()
            yi = grid_y.flatten()

            zi = idw_interpolation(x, y, z, xi, yi, power=power_idw)
            union_poly = gdf_finca.geometry.unary_union
            mask = contains(union_poly, xi, yi)
            zi[~mask] = np.nan
            grid_z = zi.reshape(grid_x.shape)

            levels = [0.0, 10.1, 20.1, 29.1, 40.1, 101.0]
            colors = ['#2e7d32', '#8bc34a', '#ffeb3b', '#f44336', '#800000']
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(levels, cmap.N)

            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm, alpha=0.9)
            gdf_finca.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.1)

            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)
            ax.axis('off')

            # --- EXPORTACIÓN A GEOPACKAGE (GPKG) PARA AVENZA ---
            # Crear un GeoDataFrame vectorizado con las geometrías de los lotes y puntos de muestreo
            gdf_puntos = gpd.GeoDataFrame(
                df_finca, 
                geometry=gpd.points_from_xy(df_finca[col_lon], df_finca[col_lat]),
                crs="EPSG:4326"
            )

            gpkg_buffer = io.BytesIO()
            # Guardar el GeoDataFrame de lotes y puntos en un paquete vectorial único
            gdf_finca.to_file(gpkg_buffer, driver="GPKG", layer="Lotes")
            
            gpkg_buffer.seek(0)

            st.pyplot(fig)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
                pdf_buffer.seek(0)
                st.download_button(
                    label="📄 Descargar Mapa PDF",
                    data=pdf_buffer,
                    file_name=f"Mapa_IDW_{finca_sel}.pdf",
                    mime="application/pdf"
                )

            with col_btn2:
                st.download_button(
                    label="🗺️ Descargar Geopackage (.gpkg para Avenza)",
                    data=gpkg_buffer,
                    file_name=f"Mapa_IDW_{finca_sel}.gpkg",
                    mime="application/geopackage+sqlite3",
                    type="primary"
                )

    except Exception as e:
        st.error(f"Error procesando la información: {e}")

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.spatial import distance
from shapely.vectorized import contains
import io

st.set_page_config(page_title="Generador IDW Cochinilla", layout="wide", initial_sidebar_state="expanded")

st.title("🐛 Generador de Mapas de Infestación por IDW")
st.markdown("Cargue los datos de muestreo de AppSheet y la capa de lotes para calcular el mapa de calor de cochinilla.")

# --- BARRA LATERAL: CARGA DE ARCHIVOS ---
st.sidebar.header("1. Carga de Datos")

file_csv = st.sidebar.file_uploader("Subir Muestreo AppSheet (CSV)", type=["csv"])
file_geojson = st.sidebar.file_uploader("Subir Lotes Finca (GeoJSON/KML)", type=["geojson", "kml", "json"])

def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((xi, yi)), np.column_stack((x, y)))
    # Evitar división por cero si el punto coincide exactamente
    dist = np.where(dist == 0, 1e-10, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    zi = np.dot(weights, z)
    return zi

if file_csv and file_geojson:
    try:
        # Carga de CSV con detección automática de separador
        try:
            df_points = pd.read_csv(file_csv, sep=';')
            if len(df_points.columns) <= 1:
                file_csv.seek(0)
                df_points = pd.read_csv(file_csv, sep=',')
        except Exception:
            file_csv.seek(0)
            df_points = pd.read_csv(file_csv, sep=',')

        # Reproyección a WGS84 (estándar para lat/lon)
        gdf_lotes = gpd.read_file(file_geojson)
        if gdf_lotes.crs is not None and gdf_lotes.crs.to_string() != "EPSG:4326":
            gdf_lotes = gdf_lotes.to_crs(epsg=4326)
        elif gdf_lotes.crs is None:
            gdf_lotes.set_crs(epsg=4326, inplace=True)

        df_points.columns = df_points.columns.str.strip()

        st.sidebar.header("2. Filtros y Parámetros IDW")

        # Filtro de Finca
        col_finca = [c for c in df_points.columns if 'FINCA' in c.upper()]
        if col_finca:
            fincas = df_points[col_finca[0]].dropna().unique()
            finca_sel = st.sidebar.selectbox("Seleccione la Finca", fincas)
            df_finca = df_points[df_points[col_finca[0]] == finca_sel].copy()
        else:
            finca_sel = "General"
            df_finca = df_points.copy()

        # Detección de Variables a interpolar
        col_brotes = [c for c in df_finca.columns if 'BROTE' in c.upper()]
        col_macollas = [c for c in df_finca.columns if 'MACOLLA' in c.upper()]

        dict_opciones = {}
        if col_brotes: dict_opciones["% Brotes Infestados"] = col_brotes[0]
        if col_macollas: dict_opciones["% Macollas Infestadas"] = col_macollas[0]

        var_label = st.sidebar.selectbox("Variable a Interpolar", list(dict_opciones.keys()))
        col_val = dict_opciones[var_label]

        # Parámetros IDW
        power_idw = st.sidebar.slider("Potencia IDW (p)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        resolution = st.sidebar.slider("Resolución Malla", min_value=50, max_value=300, value=200)

        # Limpieza de valores para análisis numérico
        for c in df_finca.columns:
            if df_finca[c].dtype == object:
                df_finca[c] = df_finca[c].astype(str).str.replace(',', '.')

        # DETECCIÓN INTELIGENTE DE COORDENADAS (Soluciona el problema de X/Y invertidos)
        col_lat, col_lon = None, None
        for col in df_finca.columns:
            try:
                temp_vals = pd.to_numeric(df_finca[col], errors='coerce').dropna()
                if not temp_vals.empty:
                    mean_val = temp_vals.mean()
                    if 10 < mean_val < 16:  # Valores en Nicaragua para Latitud (~12)
                        col_lat = col
                    elif -90 < mean_val < -80: # Valores en Nicaragua para Longitud (~-86)
                        col_lon = col
            except:
                pass
        
        # Fallback si no se detectan automáticamente
        if not col_lat: col_lat = 'Y' if 'Y' in df_finca.columns else 'Latitud'
        if not col_lon: col_lon = 'X' if 'X' in df_finca.columns else 'Longitud'

        # Limpieza de Variable Z (%)
        df_finca[col_val] = df_finca[col_val].astype(str).str.rstrip('%')
        df_finca[col_val] = pd.to_numeric(df_finca[col_val], errors='coerce')
        if df_finca[col_val].max() <= 1.0 and df_finca[col_val].max() > 0:
            df_finca[col_val] = df_finca[col_val] * 100.0

        df_finca = df_finca.dropna(subset=[col_lat, col_lon, col_val])

        # Obtener MUESTREO_DESCRIPCION
        col_desc = [c for c in df_finca.columns if 'MUESTREO_DESCRIPCION' in c.upper() or 'MUESTREO' in c.upper()]
        desc_texto = df_finca[col_desc[0]].iloc[0] if col_desc else "MUESTREO DE CAMPO"

        # Filtrar Lotes Espaciales
        col_finca_geo = [c for c in gdf_lotes.columns if 'FINCA' in c.upper() or 'CAMPO' in c.upper()]
        if col_finca_geo and finca_sel != "General":
            gdf_finca = gdf_lotes[gdf_lotes[col_finca_geo[0]].astype(str).str.lower() == str(finca_sel).lower()]
        else:
            gdf_finca = gdf_lotes

        if gdf_finca.empty: gdf_finca = gdf_lotes

        # --- BOTÓN DE GENERACIÓN ---
        if st.sidebar.button("🚀 Generar Mapa IDW", type="primary"):
            # IMPORTANTE: x es Longitud (eje X geográfico), y es Latitud (eje Y geográfico)
            x = df_finca[col_lon].to_numpy(dtype=float)
            y = df_finca[col_lat].to_numpy(dtype=float)
            z = df_finca[col_val].to_numpy(dtype=float)

            # Creación de Malla
            xmin, ymin, xmax, ymax = gdf_finca.total_bounds
            dx = (xmax - xmin) * 0.03
            dy = (ymax - ymin) * 0.03
            grid_x, grid_y = np.meshgrid(
                np.linspace(xmin - dx, xmax + dx, resolution),
                np.linspace(ymin - dy, ymax + dy, resolution)
            )

            xi = grid_x.flatten()
            yi = grid_y.flatten()

            # IDW y Máscara de Polígonos
            zi = idw_interpolation(x, y, z, xi, yi, power=power_idw)
            union_poly = gdf_finca.geometry.unary_union
            mask = contains(union_poly, xi, yi)
            zi[~mask] = np.nan
            grid_z = zi.reshape(grid_x.shape)

            # --- PALETA DE COLORES Y RANGOS ESTRICTOS ---
            # 0=Azul, 1-10=Verde, 11-29=Amarillo, 30-60=Rojo Claro, 61-100=Rojo Intenso
            levels = [0.0, 1.0, 11.0, 30.0, 61.0, 101.0]
            colors = ['#1f77b4', '#2ca02c', '#ffeb3b', '#ff6b6b', '#b20000']
            
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(levels, cmap.N)

            # --- RENDERIZADO DEL MAPA ---
            fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)

            # Usamos contourf con norm y cmap para forzar los cortes matemáticos
            contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm, alpha=0.85, zorder=2)

            # Lotes (Bordes)
            gdf_finca.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.1, zorder=3)

            # Etiquetas de Nombre de Lotes en el Centro
            col_lote_nombre = [c for c in gdf_finca.columns if c.upper() in ['CAMPO', 'LOTE', 'CODIGO_CAM', 'NOMBRE']]
            if col_lote_nombre:
                for _, row in gdf_finca.iterrows():
                    centroid = row.geometry.centroid
                    nombre_lote = str(row[col_lote_nombre[0]])
                    ax.text(centroid.x, centroid.y, nombre_lote, fontsize=8.5, fontweight='bold',
                            ha='center', va='center', color='black',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'),
                            zorder=5)

            # Puntos Muestreados Reales
            ax.scatter(x, y, c='blue', edgecolors='white', linewidth=0.7, s=45, zorder=6)

            # Encuadre y márgenes
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)

            # Leyenda Discreta
            legend_patches = [
                mpatches.Patch(color='#1f77b4', label='0% - Nulo'),
                mpatches.Patch(color='#2ca02c', label='1 - 10% - Leve'),
                mpatches.Patch(color='#ffeb3b', label='11 - 29% - Medio'),
                mpatches.Patch(color='#ff6b6b', label='30 - 60% - Alto'),
                mpatches.Patch(color='#b20000', label='61 - 100% - Muy Alto'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markeredgecolor='white', markersize=8, label='Puntos Muestreo')
            ]
            ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', title="Severidad IDW", title_fontsize='10')

            # Título dinámico
            ax.set_title(f"MAPA DE INTERPOLACIÓN IDW - COCHINILLA\n{var_label.upper()} | FINCA: {str(finca_sel).upper()}\n[{desc_texto}]", fontsize=11, fontweight='bold', pad=12)
            ax.axis('off')

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

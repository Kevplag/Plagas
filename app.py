import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.set_page_config(page_title="Generador IDW Cochinilla", layout="wide")

st.title("🐛 Generador de Mapas de Infestación de Cochinilla (IDW)")

# --- Cargar Datos ---
@st.cache_data
def load_data():
    # Cargar puntos desde CSV de AppSheet y capas vectoriales
    df_points = pd.read_csv("muestreo_cochinilla.csv")
    gdf_lotes = gpd.read_file("lotes_finca.geojson")
    return df_points, gdf_lotes

# --- Función IDW ---
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((xi, yi)), np.column_stack((x, y)))
    dist = np.where(dist == 0, 1e-10, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    zi = np.dot(weights, z)
    return zi

st.info("Sube tus archivos de muestreo CSV y GeoJSON de lotes para generar la interpolación IDW.")

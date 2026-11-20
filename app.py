"""
Generador de Mapas de Infestación por IDW - Cochinilla
-------------------------------------------------------
- Lee la capa de lotes de TODO el ingenio desde la carpeta `capas/` del repositorio
  (shapefile, GeoPackage, GeoJSON o shapefile en .zip). Opcionalmente se puede subir otra.
- Interpola en metros (UTM) y recorta a los lotes de la finca.
- Exporta un PDF GEORREFERENCIADO (Geospatial PDF, ISO 32000) listo para Avenza Maps,
  y un GeoTIFF a color (RGBA) como alternativa.
"""
import io
import os
import glob
import tempfile
import zipfile
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import distance
from pyproj import Transformer
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DictionaryObject, FloatObject, NameObject,
                           NumberObject, TextStringObject)

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    HAY_RASTERIO = True
except ImportError:
    HAY_RASTERIO = False

try:
    import contextily as ctx
    HAY_CTX = True
except ImportError:
    HAY_CTX = False

CARPETA_APP = os.path.dirname(os.path.abspath(__file__))
CARPETAS_CAPAS = [os.path.join(CARPETA_APP, "capas"), CARPETA_APP]   # capas/ y raíz del repo
EXT_CAPAS = (".shp", ".gpkg", ".geojson", ".zip", ".kml")

# Clases de infestación (cortes en .5 para que coincidan con la etiqueta redondeada)
LEVELS = [0.0, 10.5, 20.5, 29.5, 40.5, 100.0001]
COLORS = ['#2e7d32', '#8bc34a', '#ffeb3b', '#f44336', '#800000']
LABELS = ['0% - 10%', '11% - 20%', '21% - 29%', '30% - 40%', '41% - 100%']
CMAP = mcolors.ListedColormap(COLORS)
NORM = mcolors.BoundaryNorm(LEVELS, CMAP.N)

st.set_page_config(page_title="Generador IDW Cochinilla", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🐛 Generador de Mapas de Infestación por IDW")
st.markdown("Cargue los datos de muestreo de AppSheet. La capa de lotes del ingenio "
            "se lee automáticamente desde el repositorio.")


# ======================================================================
# Utilidades
# ======================================================================
def normalizar(txt):
    """Minúsculas, sin acentos ni espacios extra (para comparar nombres de finca)."""
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return " ".join(txt.lower().split())


def a_numero(serie):
    """Convierte texto con coma decimal o símbolo % a número."""
    s = serie.astype(str).str.strip().str.rstrip('%').str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def hojas_excel(data_bytes):
    return pd.ExcelFile(io.BytesIO(data_bytes)).sheet_names


def leer_excel(data_bytes, hoja):
    df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=hoja, dtype=str)
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]        # columnas vacías de Excel
    return df.dropna(how="all")                                   # filas vacías


def leer_csv(data_bytes):
    texto = data_bytes.decode('utf-8-sig', errors='replace')
    primera = texto.splitlines()[0] if texto else ""
    sep = ';' if primera.count(';') > primera.count(',') else ','
    return pd.read_csv(io.StringIO(texto), sep=sep, dtype=str)


def detectar_coord(df, nombres, vmin, vmax, excluir=()):
    """Primero por nombre de columna; si no, por rango (TODOS los valores en rango)."""
    for c in df.columns:
        if c.strip().upper() in nombres and c not in excluir:
            return c
    for c in df.columns:
        if c in excluir:
            continue
        v = a_numero(df[c])
        if v.notna().mean() > 0.9 and v.dropna().between(vmin, vmax).all():
            return c
    return None


def separar_latlong(df):
    """AppSheet exporta columnas tipo LatLong como '12.345, -86.123' en una sola columna."""
    patron = r'^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$'
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) and s.str.match(patron).mean() > 0.9:
            partes = df[c].astype(str).str.split(',', n=1, expand=True)
            df['__LAT'] = pd.to_numeric(partes[0], errors='coerce')
            df['__LON'] = pd.to_numeric(partes[1], errors='coerce')
            return '__LAT', '__LON'
    return None, None


@st.cache_data(show_spinner="Leyendo capa de lotes...")
def leer_capa(ruta):
    gdf = gpd.read_file(ruta)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def opciones_capa(ruta):
    """Devuelve las capas legibles de un archivo. Para un .zip, cada .shp/.gpkg/.geojson que contenga
    (aunque esté dentro de subcarpetas)."""
    if not ruta.lower().endswith(".zip"):
        return [ruta]
    with zipfile.ZipFile(ruta) as zf:
        internos = [n for n in zf.namelist()
                    if n.lower().endswith((".shp", ".gpkg", ".geojson")) and not n.startswith("__MACOSX")]
    return [f"zip://{ruta}!{n}" for n in internos]


def nombre_capa(ruta):
    if ruta.startswith("zip://"):
        zip_path, interno = ruta[6:].split("!", 1)
        return f"{os.path.basename(zip_path)} → {interno}"
    return os.path.basename(ruta)


def leer_capa_subida(archivo):
    sufijo = os.path.splitext(archivo.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
        tmp.write(archivo.getvalue())
    opciones = opciones_capa(tmp.name)
    if not opciones:
        raise ValueError("El .zip no contiene ningún .shp, .gpkg ni .geojson.")
    return leer_capa(opciones[0])


def adivinar_columna(columnas, preferidas):
    for p in preferidas:
        for c in columnas:
            if c.upper() == p:
                return c
    for p in preferidas:
        for c in columnas:
            if p in c.upper():
                return c
    return columnas[0] if columnas else None


def idw(xs, ys, zs, xi, yi, power=2.0, radio=None, bloque=20000):
    """IDW por bloques (no satura la memoria). radio en metros: fuera de él -> NaN."""
    out = np.full(xi.shape, np.nan)
    pts = np.column_stack((xs, ys))
    for i in range(0, len(xi), bloque):
        q = np.column_stack((xi[i:i + bloque], yi[i:i + bloque]))
        d = np.maximum(distance.cdist(q, pts), 1e-6)
        w = 1.0 / d ** power
        v = (w @ zs) / w.sum(axis=1)
        if radio:
            v[d.min(axis=1) > radio] = np.nan
        out[i:i + bloque] = v
    return out


def redondear(v):
    return int(np.floor(v + 0.5))


def esquina_box(esquina, w, h, margen=0.015):
    """Caja (x0, y0, x1, y1) en fracción de ejes para una esquina dada."""
    x0 = margen if 'left' in esquina else 1 - margen - w
    y0 = margen if 'lower' in esquina else 1 - margen - h
    return (x0, y0, x0 + w, y0 + h)


def solape(box_frac, vista, union, pts_xy):
    """Área de lotes + puntos que quedarían tapados por una caja (en fracción de ejes)."""
    vx0, vy0, vx1, vy1 = vista
    bx0 = vx0 + box_frac[0] * (vx1 - vx0)
    bx1 = vx0 + box_frac[2] * (vx1 - vx0)
    by0 = vy0 + box_frac[1] * (vy1 - vy0)
    by1 = vy0 + box_frac[3] * (vy1 - vy0)
    caja = shapely.box(bx0, by0, bx1, by1)
    area = union.intersection(caja).area / max(caja.area, 1)
    n_pts = sum(1 for px, py in pts_xy if bx0 <= px <= bx1 and by0 <= py <= by1)
    return area + 2.0 * n_pts


def mejor_esquina(candidatas, w, h, vista, union, pts_xy):
    return min(candidatas, key=lambda e: solape(esquina_box(e, w, h), vista, union, pts_xy))


def barra_escala(ax, esquina, vista):
    vx0, vy0, vx1, vy1 = vista
    ancho = vx1 - vx0
    largo = max([c for c in (25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000)
                 if c <= ancho / 5] or [25])
    w_frac = largo / ancho
    bx0, by0, _, _ = esquina_box(esquina, w_frac + 0.04, 0.06)
    x0 = vx0 + (bx0 + 0.02) * ancho
    y0 = vy0 + (by0 + 0.018) * (vy1 - vy0)
    h = (vy1 - vy0) * 0.009
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - ancho * 0.015, y0 - h * 1.6), largo + ancho * 0.03, h * 6.2,
        boxstyle='round,pad=0', fc='white', ec='black', lw=0.6, alpha=0.9, zorder=8))
    for i in range(4):
        ax.add_patch(mpatches.Rectangle((x0 + i * largo / 4, y0), largo / 4, h,
                                        fc='black' if i % 2 == 0 else 'white', ec='black', lw=0.6, zorder=9))
    etiqueta = f"{largo / 1000:g} km" if largo >= 1000 else f"{largo} m"
    for xx, t in [(x0, "0"), (x0 + largo / 2, f"{largo / 2:g}"), (x0 + largo, etiqueta)]:
        ax.text(xx, y0 + h * 1.6, t, fontsize=6, ha='center', va='bottom', zorder=9)


def rosa_vientos(fig, ax, esquina, tam_frac):
    """Rosa de los vientos N-E-S-W (como la de ArcGIS) en un eje pequeño sobre el mapa."""
    pos = ax.get_position()
    w_fig, h_fig = fig.get_size_inches()
    lado_in = tam_frac * pos.height * h_fig
    w = lado_in / w_fig
    h = lado_in / h_fig
    m = 0.015
    x0 = pos.x0 + m * pos.width if 'left' in esquina else pos.x1 - m * pos.width - w
    y0 = pos.y0 + m * pos.height if 'lower' in esquina else pos.y1 - m * pos.height - h
    axr = fig.add_axes([x0, y0, w, h], zorder=20)
    axr.set_xlim(-1.45, 1.45); axr.set_ylim(-1.45, 1.45)
    axr.set_aspect('equal'); axr.axis('off')
    axr.add_patch(mpatches.Circle((0, 0), 1.42, fc='white', ec='none', alpha=0.75))
    axr.add_patch(mpatches.Circle((0, 0), 0.62, fc='none', ec='black', lw=0.6))

    def brazo(ang, largo, ancho, oscuro):
        a = np.radians(ang)
        tip = (largo * np.cos(a), largo * np.sin(a))
        izq = (ancho * np.cos(a + np.pi / 2), ancho * np.sin(a + np.pi / 2))
        der = (ancho * np.cos(a - np.pi / 2), ancho * np.sin(a - np.pi / 2))
        axr.add_patch(mpatches.Polygon([(0, 0), tip, izq], fc=oscuro, ec='black', lw=0.5))
        axr.add_patch(mpatches.Polygon([(0, 0), tip, der], fc='white', ec='black', lw=0.5))

    for ang in (45, 135, 225, 315):
        brazo(ang, 0.62, 0.12, '#555555')
    for ang in (90, 0, 270, 180):
        brazo(ang, 1.0, 0.17, 'black')
    for ang, letra in [(90, 'N'), (0, 'E'), (270, 'S'), (180, 'W')]:
        a = np.radians(ang)
        axr.text(1.24 * np.cos(a), 1.24 * np.sin(a), letra, ha='center', va='center',
                 fontsize=8 if letra == 'N' else 6.5, fontweight='bold')
    return axr


def fmt_dms(valor, pos_letra, neg_letra):
    letra = pos_letra if valor >= 0 else neg_letra
    v = abs(valor)
    g = int(v)
    mnt = int((v - g) * 60)
    s = round(((v - g) * 60 - mnt) * 60)
    if s == 60:
        s, mnt = 0, mnt + 1
    if mnt == 60:
        mnt, g = 0, g + 1
    return f"{g}°{mnt:02d}'{s:02d}\"{letra}"


def graticula(ax, utm, vista):
    """Coordenadas geográficas (grados-minutos-segundos) en los 4 bordes del marco."""
    vx0, vy0, vx1, vy1 = vista
    a_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    a_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    lons, lats = a_geo.transform([vx0, vx1, vx0, vx1], [vy0, vy0, vy1, vy1])
    lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)

    def paso(span):
        for seg in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
            if span / (seg / 3600) <= 5:
                return seg / 3600
        return 2.0

    p_lon, p_lat = paso(lon1 - lon0), paso(lat1 - lat0)
    meridianos = np.arange(np.ceil(lon0 / p_lon) * p_lon, lon1, p_lon)
    paralelos = np.arange(np.ceil(lat0 / p_lat) * p_lat, lat1, p_lat)
    m = np.linspace(lat0 - p_lat, lat1 + p_lat, 60)
    n = np.linspace(lon0 - p_lon, lon1 + p_lon, 60)

    abajo, arriba, izq, der = [], [], [], []
    for lon in meridianos:
        xs, ys = a_utm.transform(np.full_like(m, lon), m)
        lab = fmt_dms(lon, 'E', 'W')
        for lista, yy in ((abajo, vy0), (arriba, vy1)):
            xx = np.interp(yy, ys, xs)
            if vx0 < xx < vx1:
                lista.append((xx, lab))
    for lat in paralelos:
        xs, ys = a_utm.transform(n, np.full_like(n, lat))
        lab = fmt_dms(lat, 'N', 'S')
        for lista, xx in ((izq, vx0), (der, vx1)):
            yy = np.interp(xx, xs, ys)
            if vy0 < yy < vy1:
                lista.append((yy, lab))

    estilo = dict(direction='in', length=4, width=0.6, labelsize=6.5, pad=2)
    ax.set_xticks([v for v, _ in abajo], [t for _, t in abajo])
    ax.set_yticks([v for v, _ in izq], [t for _, t in izq], rotation=90, va='center')
    ax.tick_params(**estilo)
    sx = ax.secondary_xaxis('top')
    sx.set_xticks([v for v, _ in arriba], [t for _, t in arriba])
    sx.tick_params(**estilo)
    sy = ax.secondary_yaxis('right')
    sy.set_yticks([v for v, _ in der], [t for _, t in der], rotation=270, va='center')
    sy.tick_params(**estilo)
    ax.set_xlabel(''); ax.set_ylabel('')


def angulo_lote(geom):
    """Ángulo del eje largo del lote, para girar su etiqueta (como en ArcGIS)."""
    try:
        r = geom.minimum_rotated_rectangle
        c = np.array(r.exterior.coords)[:3]
        lados = [c[1] - c[0], c[2] - c[1]]
        v = max(lados, key=lambda d: np.hypot(*d))
        ang = np.degrees(np.arctan2(v[1], v[0]))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        return ang
    except Exception:
        return 0.0


MAPAS_BASE = {
    "Topográfico (Esri)": "Esri.WorldTopoMap",
    "Satélite (Esri)": "Esri.WorldImagery",
    "Calles (OpenStreetMap)": "OpenStreetMap.Mapnik",
    "Claro (CARTO)": "CartoDB.Positron",
    "Ninguno": None,
}


def agregar_mapa_base(ax, utm, nombre):
    clave = MAPAS_BASE.get(nombre)
    if not clave:
        return None
    if not HAY_CTX:
        return "No está instalado 'contextily': el mapa se generó sin mapa base."
    try:
        prov = ctx.providers
        for parte in clave.split('.'):
            prov = prov[parte]
        ctx.add_basemap(ax, crs=utm.to_string(), source=prov, attribution_size=5,
                        zorder=0, reset_extent=True)
        return None
    except Exception as e:
        return f"No se pudo descargar el mapa base ({e}). El mapa se generó sin él."


def geopdf(pdf_bytes, bbox_pts, esquinas_latlon, epsg, wkt):
    """
    Agrega georreferenciación ISO 32000 (Adobe Geospatial PDF) a la página 1.
    bbox_pts: (x0, y0, x1, y1) del área del mapa en puntos PDF.
    esquinas_latlon: [(lat,lon) sup-izq, inf-izq, inf-der, sup-der].
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter(clone_from=reader)
    page = writer.pages[0]

    gcs = DictionaryObject({
        NameObject("/Type"): NameObject("/PROJCS"),
        NameObject("/EPSG"): NumberObject(int(epsg)),
        NameObject("/WKT"): TextStringObject(wkt),
    })
    gpts = ArrayObject()
    for lat, lon in esquinas_latlon:
        gpts.extend([FloatObject(round(lat, 9)), FloatObject(round(lon, 9))])
    unit_sq = ArrayObject([FloatObject(v) for v in (0, 1, 0, 0, 1, 0, 1, 1)])
    measure = DictionaryObject({
        NameObject("/Type"): NameObject("/Measure"),
        NameObject("/Subtype"): NameObject("/GEO"),
        NameObject("/Bounds"): unit_sq,
        NameObject("/GPTS"): gpts,
        NameObject("/LPTS"): ArrayObject(list(unit_sq)),
        NameObject("/GCS"): writer._add_object(gcs),
        NameObject("/PDU"): ArrayObject([NameObject("/M"), NameObject("/SQM"), NameObject("/DEG")]),
    })
    viewport = DictionaryObject({
        NameObject("/Type"): NameObject("/Viewport"),
        NameObject("/Name"): TextStringObject("Mapa IDW"),
        NameObject("/BBox"): ArrayObject([FloatObject(round(v, 4)) for v in bbox_pts]),
        NameObject("/Measure"): writer._add_object(measure),
    })
    page[NameObject("/VP")] = ArrayObject([writer._add_object(viewport)])

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def geotiff_rgba(grid_z, bounds, crs):
    """GeoTIFF a color (RGBA 8 bits), transparente fuera de los lotes."""
    h, w = grid_z.shape
    z = np.flipud(grid_z)                       # fila 0 = norte
    rgba = (CMAP(NORM(np.nan_to_num(z, nan=0))) * 255).astype(np.uint8)
    rgba[np.isnan(z)] = 0
    transform = from_bounds(*bounds, w, h)
    with MemoryFile() as mem:
        with mem.open(driver='GTiff', height=h, width=w, count=4, dtype='uint8',
                      crs=crs, transform=transform, photometric='RGB', alpha='YES',
                      compress='deflate') as dst:
            for i in range(4):
                dst.write(rgba[:, :, i], i + 1)
        return mem.read()


# ======================================================================
# 1. Carga de datos
# ======================================================================
st.sidebar.header("1. Carga de Datos")
file_csv = st.sidebar.file_uploader("Subir Muestreo AppSheet (CSV o Excel)",
                                    type=["csv", "xlsx", "xlsm", "xls"])
hoja_excel = None
if file_csv and not file_csv.name.lower().endswith(".csv"):
    try:
        hojas = hojas_excel(file_csv.getvalue())
        hoja_excel = st.sidebar.selectbox("Hoja del Excel", hojas) if len(hojas) > 1 else hojas[0]
    except Exception as e:
        st.error("No se pudo abrir el archivo Excel.")
        st.exception(e)
        st.stop()

capas_repo = []
for carpeta in CARPETAS_CAPAS:
    for p in sorted(glob.glob(os.path.join(carpeta, "*"))):
        if p.lower().endswith(EXT_CAPAS):
            try:
                capas_repo.extend(opciones_capa(p))
            except zipfile.BadZipFile:
                pass
with st.sidebar.expander("Capa de lotes", expanded=not capas_repo):
    if capas_repo:
        capa_sel = st.selectbox("Capa del repositorio", capas_repo, format_func=nombre_capa)
    else:
        capa_sel = None
        st.warning("No hay capas en el repositorio (raíz o carpeta `capas/`).")
    capa_subida = st.file_uploader("O subir otra capa (opcional)",
                                   type=["zip", "gpkg", "geojson", "json", "kml"],
                                   help="Shapefile comprimido en .zip con .shp, .shx, .dbf y .prj")

if not file_csv:
    st.info("👈 Sube el archivo de muestreo (CSV o Excel) para comenzar.")
    st.stop()
if not (capa_subida or capa_sel):
    st.error("No hay capa de lotes: agrégala a la carpeta `capas/` o súbela en la barra lateral.")
    st.stop()

try:
    if hoja_excel is not None:
        df_points = leer_excel(file_csv.getvalue(), hoja_excel)
    else:
        df_points = leer_csv(file_csv.getvalue())
    df_points.columns = df_points.columns.str.strip()
    gdf_lotes = leer_capa_subida(capa_subida) if capa_subida else leer_capa(capa_sel)
except Exception as e:
    st.error("No se pudieron leer los archivos.")
    st.exception(e)
    st.stop()

# ======================================================================
# 2. Filtros y parámetros
# ======================================================================
st.sidebar.header("2. Filtros y Parámetros IDW")

col_finca = next((c for c in df_points.columns if 'FINCA' in c.upper()), None)
if col_finca:
    fincas = sorted(df_points[col_finca].dropna().unique())
    finca_sel = st.sidebar.selectbox("Seleccione la Finca", fincas)
    df_finca = df_points[df_points[col_finca] == finca_sel].copy()
else:
    finca_sel = "General"
    df_finca = df_points.copy()

dict_opciones = {}
c_brote = next((c for c in df_finca.columns if 'BROTE' in c.upper()), None)
c_macolla = next((c for c in df_finca.columns if 'MACOLLA' in c.upper()), None)
if c_brote:
    dict_opciones["% Brotes Infestados"] = c_brote
if c_macolla:
    dict_opciones["% Macollas Infestadas"] = c_macolla
if not dict_opciones:
    st.error("El archivo de muestreo no tiene columnas de BROTES ni MACOLLAS.")
    st.stop()
var_label = st.sidebar.selectbox("Variable a Interpolar", list(dict_opciones.keys()))
col_val = dict_opciones[var_label]

# Columnas de la capa de lotes
cols_capa = [c for c in gdf_lotes.columns if c != 'geometry']
with st.sidebar.expander("Columnas de la capa de lotes"):
    col_finca_geo = st.selectbox(
        "Columna de FINCA", cols_capa,
        index=cols_capa.index(adivinar_columna(cols_capa, ['FINCA', 'HACIENDA', 'NOM_FINCA', 'NOMBRE_FIN'])))
    col_lote = st.selectbox(
        "Columna con el nombre del LOTE", cols_capa,
        index=cols_capa.index(adivinar_columna(cols_capa, ['LOTE', 'CODIGO_LOT', 'COD_LOTE', 'CAMPO', 'CODIGO_CAM', 'NOMBRE'])))

power_idw = st.sidebar.slider("Potencia IDW (p)", 1.0, 5.0, 2.0, 0.5)
resolution = st.sidebar.slider("Resolución Malla (celdas lado mayor)", 100, 600, 300, 50)
radio_max = st.sidebar.number_input(
    "Radio máximo de influencia (m) — 0 = sin límite", min_value=0, value=0, step=50,
    help="Las zonas más lejanas que este radio a cualquier punto quedan en blanco.")
mapa_base = st.sidebar.selectbox("Mapa base", list(MAPAS_BASE.keys()), index=0)
AREAS = ["Solo lotes muestreados",
         "Toda la finca (lotes sin muestreo en gris)",
         "Toda la finca (interpolar todo)"]
area_mapa = st.sidebar.radio("Área a mapear", AREAS, index=0,
                             help="El IDW solo es confiable dentro de los lotes que tienen puntos. "
                                  "Interpolar toda la finca rellena lotes sin muestreo con el valor de "
                                  "puntos lejanos.")
tolerancia_gps = st.sidebar.number_input("Tolerancia GPS para asignar puntos a un lote (m)",
                                         min_value=0, value=30, step=5)

# ---------- Coordenadas ----------
col_lat = detectar_coord(df_finca, {'LAT', 'LATITUD', 'LATITUDE', 'Y'}, 10, 16, excluir={col_val})
col_lon = detectar_coord(df_finca, {'LON', 'LONG', 'LONGITUD', 'LONGITUDE', 'X'}, -90, -80, excluir={col_val})
if not col_lat or not col_lon:
    col_lat, col_lon = separar_latlong(df_finca)
if not col_lat or not col_lon:
    st.error("No se encontraron columnas de latitud/longitud en el archivo de muestreo.")
    st.stop()
df_finca['__lat'] = a_numero(df_finca[col_lat])
df_finca['__lon'] = a_numero(df_finca[col_lon])

# ---------- Variable ----------
tenia_pct = df_finca[col_val].astype(str).str.contains('%').any()
df_finca['__val'] = a_numero(df_finca[col_val])
if not tenia_pct and 0 < df_finca['__val'].max() <= 1.0:
    df_finca['__val'] *= 100.0
df_finca = df_finca.dropna(subset=['__lat', '__lon', '__val'])
if df_finca.empty:
    st.error("No hay puntos válidos (coordenadas y valor) para esta finca.")
    st.stop()

col_desc = next((c for c in df_finca.columns if 'DESCRIP' in c.upper()), None) or \
           next((c for c in df_finca.columns if 'MUESTREO' in c.upper()), None)
desc_texto = str(df_finca[col_desc].iloc[0]) if col_desc else "MUESTREO DE CAMPO"

# ---------- Lotes de la finca ----------
gdf_pts = gpd.GeoDataFrame(df_finca, geometry=gpd.points_from_xy(df_finca['__lon'], df_finca['__lat']),
                           crs="EPSG:4326")
gdf_pts_l = gdf_pts.to_crs(gdf_lotes.crs)

gdf_finca = gdf_lotes.iloc[0:0]
if finca_sel != "General":
    gdf_finca = gdf_lotes[gdf_lotes[col_finca_geo].map(normalizar) == normalizar(finca_sel)]
if gdf_finca.empty:
    # Respaldo: buscar la(s) finca(s) donde caen los puntos
    unidos = gpd.sjoin(gdf_pts_l[['geometry']], gdf_lotes[[col_finca_geo, 'geometry']],
                       how='inner', predicate='within')
    fincas_pts = unidos[col_finca_geo].dropna().unique()
    if len(fincas_pts):
        gdf_finca = gdf_lotes[gdf_lotes[col_finca_geo].isin(fincas_pts)]
        st.warning(f"'{finca_sel}' no aparece en la columna '{col_finca_geo}' de la capa. "
                   f"Se usaron los lotes donde caen los puntos: {', '.join(map(str, fincas_pts))}.")
    else:
        st.error(f"No se encontraron lotes para '{finca_sel}' y ningún punto cae dentro de la capa. "
                 "Revisa la columna de FINCA en 'Columnas de la capa de lotes'.")
        st.stop()

# ---------- Lotes muestreados (por NOMBRE de lote, no por polígono) ----------
utm_sel = gdf_finca.estimate_utm_crs()
finca_utm = gdf_finca.to_crs(utm_sel)
pts_utm = gdf_pts.to_crs(utm_sel)[['geometry']]
cerca = gpd.sjoin_nearest(pts_utm, finca_utm[[col_lote, 'geometry']], how='inner',
                          max_distance=max(tolerancia_gps, 0.01))
nombres_muestreados = set(cerca[col_lote].dropna().map(normalizar))

# Si el muestreo trae una columna de lote/campo, también se usa
col_lote_csv = next((c for c in df_finca.columns
                     if any(k in c.upper() for k in ('LOTE', 'CAMPO')) and 'DESCRIP' not in c.upper()), None)
if col_lote_csv:
    nombres_capa = set(gdf_finca[col_lote].dropna().map(normalizar))
    nombres_muestreados |= set(df_finca[col_lote_csv].dropna().map(normalizar)) & nombres_capa

es_muestreado = gdf_finca[col_lote].map(normalizar).isin(nombres_muestreados)
if not es_muestreado.any():
    st.warning("Ningún punto cae dentro (o cerca) de un lote de la finca: se interpola toda la finca. "
               "Revisa las coordenadas o la columna de LOTE.")
    es_muestreado = pd.Series(True, index=gdf_finca.index)

gdf_muestreados = gdf_finca[es_muestreado]
if area_mapa == AREAS[0]:
    gdf_vista, gdf_interp = gdf_muestreados, gdf_muestreados
elif area_mapa == AREAS[1]:
    gdf_vista, gdf_interp = gdf_finca, gdf_muestreados
else:
    gdf_vista, gdf_interp = gdf_finca, gdf_finca

fuera = len(pts_utm) - cerca.index.nunique()
if fuera:
    st.warning(f"{fuera} punto(s) están a más de {tolerancia_gps} m de cualquier lote de la finca.")

if len(df_finca) < 15:
    st.warning(f"Solo hay {len(df_finca)} puntos de muestreo: el mapa es indicativo. "
               "Las zonas sin muestreo cercano toman el valor de los puntos más próximos.")


# ======================================================================
# 3. Generación
# ======================================================================
def generar():
    utm = gdf_vista.estimate_utm_crs()
    lotes = gdf_vista.to_crs(utm)
    lotes_interp = gdf_interp.to_crs(utm)
    lotes_sin = lotes.loc[~lotes.index.isin(lotes_interp.index)]
    pts = gdf_pts.to_crs(utm)
    x = pts.geometry.x.to_numpy()
    y = pts.geometry.y.to_numpy()
    z = pts['__val'].to_numpy(dtype=float)

    bxmin, bymin, bxmax, bymax = lotes.total_bounds
    bxmin, bymin = min(bxmin, x.min()), min(bymin, y.min())
    bxmax, bymax = max(bxmax, x.max()), max(bymax, y.max())
    m = max(bxmax - bxmin, bymax - bymin) * 0.03
    xmin, ymin, xmax, ymax = bxmin - m, bymin - m, bxmax + m, bymax + m

    # Malla de celdas cuadradas, puntos en el CENTRO de cada píxel
    celda = max(xmax - xmin, ymax - ymin) / resolution
    nx = int(np.ceil((xmax - xmin) / celda))
    ny = int(np.ceil((ymax - ymin) / celda))
    xmax, ymax = xmin + nx * celda, ymin + ny * celda
    gx, gy = np.meshgrid(xmin + (np.arange(nx) + 0.5) * celda,
                         ymin + (np.arange(ny) + 0.5) * celda)

    def unir(g):
        return g.union_all() if hasattr(g, "union_all") else g.unary_union
    union = unir(lotes)                         # todo lo visible (para ubicar leyenda, etc.)
    mask = shapely.contains_xy(unir(lotes_interp), gx.ravel(), gy.ravel())
    zi = np.full(gx.size, np.nan)
    zi[mask] = idw(x, y, z, gx.ravel()[mask], gy.ravel()[mask],
                   power=power_idw, radio=radio_max or None)
    grid_z = zi.reshape(gx.shape)

    # ---------------- Página y extensión de la vista ----------------
    horizontal = (bxmax - bxmin) >= (bymax - bymin)
    figsize = (13, 8.5) if horizontal else (8.5, 11)
    rect = [0.045, 0.035, 0.91, 0.855] if horizontal else [0.06, 0.03, 0.88, 0.885]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    # La vista se amplía para llenar todo el marco (sin bordes blancos) manteniendo escala 1:1
    relacion = (rect[2] * figsize[0]) / (rect[3] * figsize[1])
    cx, cy = (bxmin + bxmax) / 2, (bymin + bymax) / 2
    vw = (bxmax - bxmin) * 1.12
    vh = (bymax - bymin) * 1.12
    if vw / vh < relacion:
        vw = vh * relacion
    else:
        vh = vw / relacion
    vista = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)
    ax.set_xlim(vista[0], vista[2]); ax.set_ylim(vista[1], vista[3])
    ax.set_aspect('equal', adjustable='box')

    aviso_base = agregar_mapa_base(ax, utm, mapa_base)

    ax.contourf(gx, gy, np.ma.masked_invalid(grid_z), levels=LEVELS, cmap=CMAP, norm=NORM,
                alpha=0.88, zorder=2)
    if not lotes_sin.empty:
        lotes_sin.plot(ax=ax, facecolor="#d9d9d9", edgecolor="#7f7f7f", hatch='////',
                       linewidth=0.5, alpha=0.8, zorder=2.5)
    lotes.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=3)

    # Una etiqueta por lote (aunque el lote tenga varias partes), en su parte más grande,
    # girada según la forma del lote
    partes = lotes[[col_lote, 'geometry']].explode(index_parts=False).reset_index(drop=True)
    partes['__area'] = partes.geometry.area
    for nombre, grupo in partes.groupby(col_lote):
        geom = grupo.loc[grupo['__area'].idxmax(), 'geometry']
        p = geom.representative_point()
        ax.text(p.x, p.y, str(nombre), fontsize=5.5, ha='center', va='center',
                rotation=angulo_lote(geom), rotation_mode='anchor', color='black', zorder=4,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.6, edgecolor='none'))

    ax.scatter(x, y, c='#3e2723', edgecolors='white', linewidth=0.7, s=26, zorder=6)
    for px, py, pz in zip(x, y, z):
        ax.annotate(f"{redondear(pz)}%", (px, py), xytext=(4, 3), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color='black', ha='left', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='gray',
                              linewidth=0.5, alpha=0.9))

    # ---------------- Leyenda (dentro del mapa, en la mejor esquina) ----------------
    titulo_var = 'BROTES' if 'BROTE' in var_label.upper() else 'MACOLLAS'
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#3e2723',
               markeredgecolor='white', markersize=6, label='Estaciones'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='Lotes'),
        mpatches.Patch(color='none', label=f'INFESTACIÓN DE {titulo_var}'),
        mpatches.Patch(color='none', label='<RANGOS>'),
    ] + [mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.4, label=l)
         for c, l in zip(COLORS, LABELS)]
    if not lotes_sin.empty:
        handles.append(mpatches.Patch(facecolor='#d9d9d9', edgecolor='#7f7f7f', hatch='////',
                                      linewidth=0.4, label='Sin muestreo'))

    def crear_leyenda(loc):
        leg = ax.legend(handles=handles, loc=loc, title='Leyenda', title_fontsize=8.5,
                        fontsize=7, frameon=True, framealpha=0.95, facecolor='white',
                        edgecolor='black', borderpad=0.7, labelspacing=0.45, handlelength=1.6,
                        borderaxespad=0.8)
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_ha('left')
        leg._legend_box.align = "left"
        for t in leg.get_texts()[2:4]:
            t.set_fontweight('bold'); t.set_fontsize(7)
        leg.get_frame().set_linewidth(0.8)
        leg.set_zorder(15)
        return leg

    leg = crear_leyenda('upper right')
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    pts_xy = list(zip(x, y))
    esquinas = ['upper left', 'upper right', 'lower left', 'lower right']
    esq_leyenda = mejor_esquina(esquinas, bb.width, bb.height, vista, union, pts_xy)
    leg.remove()
    crear_leyenda(esq_leyenda)

    # Rosa de los vientos (arriba) y escala (abajo), en las esquinas libres
    tam_rosa = 0.13
    pos = ax.get_position()
    w_rosa = tam_rosa * pos.height * figsize[1] / (pos.width * figsize[0])
    # La rosa prefiere arriba, pero puede ir abajo si arriba tapa lotes o puntos
    libres = [e for e in esquinas if e != esq_leyenda]
    esq_rosa = min(libres, key=lambda e: solape(esquina_box(e, w_rosa, tam_rosa), vista, union, pts_xy)
                   + (0.03 if 'lower' in e else 0))
    libres = [e for e in libres if e != esq_rosa]
    esq_escala = min(libres, key=lambda e: solape(esquina_box(e, 0.25, 0.07), vista, union, pts_xy)
                     + (0.03 if 'upper' in e else 0))
    barra_escala(ax, esq_escala, vista)

    # ---------------- Coordenadas y título ----------------
    graticula(ax, utm, vista)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.annotate("MAPA DE INFESTACIÓN POR COCHINILLA HARINOSA", (0.5, 1), xycoords='axes fraction',
                xytext=(0, 30), textcoords='offset points', ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    ax.annotate(f"{var_label.upper()}  |  FINCA: {str(finca_sel).upper()}  |  {desc_texto}",
                (0.5, 1), xycoords='axes fraction', xytext=(0, 15), textcoords='offset points',
                ha='center', va='bottom', fontsize=9.5)

    # Posición real del mapa en la página (en puntos PDF), después de todo el dibujo
    fig.canvas.draw()
    rosa_vientos(fig, ax, esq_rosa, tam_rosa)
    pos = ax.get_position()
    W, H = fig.get_size_inches() * 72
    bbox_pts = (pos.x0 * W, pos.y0 * H, pos.x1 * W, pos.y1 * H)

    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=130)
    pdf = io.BytesIO()
    fig.savefig(pdf, format='pdf', dpi=250)     # SIN bbox_inches='tight' (cambiaría la posición)
    plt.close(fig)

    # ---------------- GeoPDF ----------------
    a_geo = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    esquinas_ll = []
    for ex, ey in [(vista[0], vista[3]), (vista[0], vista[1]), (vista[2], vista[1]), (vista[2], vista[3])]:
        lon, lat = a_geo.transform(ex, ey)
        esquinas_ll.append((lat, lon))
    pdf_geo = geopdf(pdf.getvalue(), bbox_pts, esquinas_ll, utm.to_epsg(), utm.to_wkt(version="WKT1_GDAL"))

    tif = None
    if HAY_RASTERIO:
        tif = geotiff_rgba(grid_z, (xmin, ymin, xmax, ymax), utm.to_wkt())

    return dict(png=png.getvalue(), pdf=pdf_geo, tif=tif, n=len(z), aviso=aviso_base,
                zmax=float(z.max()), zmean=float(z.mean()), epsg=utm.to_epsg())


firma = (finca_sel, var_label, power_idw, resolution, radio_max, area_mapa, tolerancia_gps, mapa_base,
         col_lote, col_finca_geo, file_csv.name, hoja_excel, capa_subida.name if capa_subida else capa_sel)

if st.sidebar.button("🚀 Generar Mapa IDW", type="primary"):
    try:
        with st.spinner("Calculando interpolación..."):
            st.session_state['resultado'] = generar()
            st.session_state['firma'] = firma
            st.session_state['nombre'] = f"Mapa_IDW_{var_label.replace('% ', '').replace(' ', '_')}_{finca_sel}"
    except Exception as e:
        st.error("Error al generar el mapa.")
        st.exception(e)

# ======================================================================
# 4. Resultados (persisten al descargar gracias a session_state)
# ======================================================================
res = st.session_state.get('resultado')
if res:
    nombre = st.session_state['nombre']
    if st.session_state.get('firma') != firma:
        st.info("Cambiaste la finca o los parámetros: presiona **Generar Mapa IDW** para actualizar.")
    if res.get('aviso'):
        st.warning(res['aviso'])
    col_map, col_stats = st.columns([3, 1])
    with col_map:
        st.image(res['png'])
        b1, b2 = st.columns(2)
        with b1:
            st.download_button("🗺️ Descargar PDF georreferenciado (Avenza Maps)", data=res['pdf'],
                               file_name=f"{nombre}.pdf", mime="application/pdf", type="primary")
        with b2:
            if res['tif'] is not None:
                st.download_button("🛰️ Descargar GeoTIFF a color", data=res['tif'],
                                   file_name=f"{nombre}.tif", mime="image/tiff")
    with col_stats:
        st.subheader("📊 Resumen")
        st.metric("Puntos Evaluados", res['n'])
        st.metric("Infestación Máxima", f"{res['zmax']:.1f}%")
        st.metric("Infestación Promedio", f"{res['zmean']:.1f}%")
        st.caption(f"Proyección del mapa: EPSG:{res['epsg']} (UTM)")

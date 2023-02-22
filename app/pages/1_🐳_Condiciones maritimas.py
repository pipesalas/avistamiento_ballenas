import pandas as pd
import numpy as np
import geopandas as gpd
from config import DATES
import streamlit as st
import pydeck as pdk



st.header('Visualizador de las condiciones marítimas')

diccionario_meses = {'enero 2023': '2023-01-01_to_2023-02-01',
                     'febrero 2023': '2023-02-01_to_2023-03-01',}

mes_seleccionado = st.selectbox('Seleccionamos un mes', diccionario_meses.keys())

paths_temp = [f'../data/temperature_polygons_{diccionario_meses.get(mes_seleccionado)}.csv',
            f'../data/temperature_points_{diccionario_meses.get(mes_seleccionado)}.csv']


paths_chlor = [f'../data/chlorophyll_polygons_{diccionario_meses.get(mes_seleccionado)}.csv',
            f'../data/chlorophyll_points_{diccionario_meses.get(mes_seleccionado)}.csv']

df_temp = pd.read_csv(paths_chlor[0])
gdf_temp = gpd.read_file(f'../data/temperature_polygons_{diccionario_meses.get(mes_seleccionado)}.json')

# Custom color scale
COLOR_RANGE = [
    [65, 182, 196],
    [127, 205, 187],
    [199, 233, 180],
    [237, 248, 177],
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
    [128, 0, 38],
]

BREAKS = [9, 9.4, 9.8, 10.2, 10.6, 11, 11.4, 11.8, 12.2, 12.6, 13, 13.4, 13.8]


def color_scale(val):
    for i, b in enumerate(BREAKS):
        if val < b:
            return COLOR_RANGE[i]
    return COLOR_RANGE[i]

gdf_temp["fill_color"] = gdf_temp["temp"].apply(lambda row: color_scale(row))


st.header('Temperature on sea surface')
view_state = pdk.ViewState(
    **{"latitude": -40, "longitude": -73.8, "zoom": 9.5, "maxZoom": 16, "pitch": 20, "bearing": 0}
)

polygon_layer = pdk.Layer(
    "PolygonLayer",
    data=gdf_temp,
    #id="temp",
    opacity=0.2,
    stroked=False,
    get_polygon="geometry.coordinates",
    filled=True,
    #wireframe=True,
    get_fill_color="fill_color",
    get_line_color=[255, 255, 255],
    #auto_highlight=True,
    pickable=True,
)

tooltip = {"html": "<b>Value per Square Meter:</b> {temp}"}

r = pdk.Deck(
    polygon_layer,
    initial_view_state=view_state,
    map_style=pdk.map_styles.LIGHT,
    tooltip=tooltip,
)
st.pydeck_chart(r)


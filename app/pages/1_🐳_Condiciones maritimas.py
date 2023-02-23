import pandas as pd
import numpy as np
import geopandas as gpd
from config import DATES
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk

import os


def main():

    st.header('Visualizador de las condiciones marítimas :ocean:')

    diccionario_meses = {
                        'septiembre 2023': '2022-09-01_to_2022-10-01',
                        'octubre 2023': '2022-10-01_to_2022-11-01',
                        'noviembre 2023': '2022-11-01_to_2022-12-01',
                        'diciembre 2023': '2022-12-01_to_2023-01-01',
                        'enero 2023': '2023-01-01_to_2023-02-01',
                        'febrero 2023': '2023-02-01_to_2023-03-01',}

    mes_seleccionado = st.selectbox('Seleccionamos un mes', diccionario_meses.keys())


    paths_temp = [f'app/data/temperature_polygons_{diccionario_meses.get(mes_seleccionado)}.csv',
                f'app/data/temperature_points_{diccionario_meses.get(mes_seleccionado)}.csv']


    paths_chlor = [f'app/data/chlorophyll_polygons_{diccionario_meses.get(mes_seleccionado)}.csv',
                f'app/data/chlorophyll_points_{diccionario_meses.get(mes_seleccionado)}.csv']



    df_temp = pd.read_csv(paths_temp[1])
    gdf_temp = gpd.read_file(f'app/data/temperature_polygons_{diccionario_meses.get(mes_seleccionado)}.json')
    plot_temperature(gdf_temp)

    df_chlor = pd.read_csv(paths_chlor[1])
    gdf_chlor = gpd.read_file(f'app/data/chlorophyll_polygons_{diccionario_meses.get(mes_seleccionado)}.json')

    plot_chlorophyll(gdf_chlor)



def plot_chlorophyll(gdf):

    # Custom color scale
    COLOR_RANGE = [[230, 250, 250], #E6FAFA
                   [93, 229, 230], #C1E5E6
                   [157, 208, 212], #9DD0D4
                   [117, 187, 193], #75BBC1
                   [75, 167, 175], #4BA7AF
                   [0, 147, 156], #00939C
                   [16, 129, 136], #108188
                   [14, 112, 119]] #0E7077

    BREAKS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]


    def color_scale(val):
        for i, b in enumerate(BREAKS):
            if val < b:
                return COLOR_RANGE[i]
        return COLOR_RANGE[i]

    gdf["fill_color"] = gdf["chlor"].apply(lambda row: color_scale(row))


    st.header('Chlorophyll on sea surface 🌡')
    view_state = pdk.ViewState(
        **{"latitude": -40, "longitude": -73.8, "zoom": 9.5, "maxZoom": 16, "pitch": 20, "bearing": 0}
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf,
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

    tooltip = {"html": "<b>Value per Square Meter:</b> {chlor}"}

    r = pdk.Deck(
        polygon_layer,
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip=tooltip,
    )
    st.pydeck_chart(r)





def plot_temperature(gdf_temp):

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


    st.header('Temperature on sea surface 🌡')
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


main()
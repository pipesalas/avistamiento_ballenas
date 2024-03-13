import pandas as pd
import numpy as np
import geopandas as gpd
from config import DATES
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk
import os
import requests
from netCDF4 import Dataset


def main():

    st.set_page_config(
    page_title="Condiciones maritimas",
    page_icon="🌊",
    layout="wide",
    menu_items={
        'About': "# Datos de avistamiento marinos realizados por *vuelvealoceano*",
        #'Vuelve Al Oceano': 'https://www.vuelvealoceano.cl',
    }
    )

    st.title('🐋 Avistamiento de mamiferos marinos :ocean:')

    #st.('''Para visualizar las condiciones maritimas usamos google earth engine, sacamos la MEDIANA del mes, blabla''' )
    st.markdown('**Bienvenido a nuestra aplicación de avistamiento de observaciones de Ballenas en Chile**')
    


    

    # cargamos datos
    df_avistamientos = load_datos_avistamientos()
    ruta = load_ruta()
    chlorophyll = load_chlorophyll()
    temperature = load_temperature()


    with st.expander('Datos brutos', expanded=True):
        st.write(df_avistamientos.head())
        st.write(chlorophyll.head())
        st.write(temperature.query('time<"2023-01-02"').sample(15))
        st.write(temperature.dtypes)

    # Create bins for temperature
    bins = np.linspace(temperature['temperature'].min(), temperature['temperature'].max(), 10)
    labels = range(1, len(bins))
    temperature['temp_bin'] = pd.cut(temperature['temperature'], bins=bins, labels=labels)

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=-39.5,
            longitude=-73.5,
            zoom=8,
            pitch=50,
        ),
        layers=[
            # pdk.Layer(
            # 'HeatmapLayer',
            # data=temperature.query('time<"2023-01-02"')[['lon','lat', 'temperature']],
            # get_position='[lon, lat]',
            # get_weight='temperature',
            # opacity=0.1,
            # threshold=temperature.query('time<"2023-01-02"').temperature.mean(),
            # intensity=10,
            # aggregation='"MEAN"',
            # ),
            pdk.Layer(
                'PolygonLayer',
                data=temperature.query('time<"2023-01-02"')[['lon', 'lat', 'geometry', 'temperature']],
                get_polygon='geometry.coordinates',
                get_elevation='temperature',
                get_fill_color=[180, 0, 200, 140],
                get_line_color=[255, 255, 255],
                line_width_min_pixels=2,
                opacity=0.5,
            ),
        ],
    ))
    # Set the initial view
    # view_state = pdk.ViewState(
    #     latitude=df_avistamientos['lat'].mean(),
    #     longitude=df_avistamientos['lon'].mean(),
    #     zoom=6,
    #     min_zoom=5,
    #     max_zoom=15,
    #     pitch=40.5,
    #     bearing=-27.36)

    # # Create a layer for temperature heatmap
    # temperature_layer = pdk.Layer(
    #     'HeatmapLayer',
    #     data=temperature,
    #     opacity=0.9,
    #     get_position=['longitude', 'latitude'],
    #     threshold=0.3,
    #     get_weight='temperature',
    #     aggregation='"MEAN"'
    # )

    # # Create a layer for sightings
    # sightings_layer = pdk.Layer(
    #     'ScatterplotLayer',
    #     data=df_avistamientos,
    #     get_position=['lon', 'lat'],
    #     get_color=[200, 30, 0, 160],
    #     get_radius=200,
    # )

    # # Combine all layers
    # layers = [temperature_layer, sightings_layer]

    # # Render the map
    # st.pydeck_chart(pdk.Deck(
    #     map_style='mapbox://styles/mapbox/light-v9',
    #     initial_view_state=view_state,
    #     layers=layers,
    # ))


@st.cache_data()
def load_temperature():

    temperature = pd.read_pickle('app/data/temperature.pkl')
    min_depth = temperature.depth.min()
    temperature = temperature.query('temperature > 0 and depth == @min_depth')
    temperature['temperature'] = temperature['temperature'].astype(float)
    temperature.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    temperature.dropna(how='any', inplace=True)
    # Define the grid size
    grid_size = 0.08

    # Create a square geometry around each coordinate
    temperature['geometry'] = temperature.apply(lambda row: MultiPoint([(row['lon'] - grid_size / 2, row['lat'] - grid_size / 2), 
                                                                        (row['lon'] + grid_size / 2, row['lat'] - grid_size / 2), 
                                                                        (row['lon'] + grid_size / 2, row['lat'] + grid_size / 2), 
                                                                        (row['lon'] - grid_size / 2, row['lat'] + grid_size / 2)]).convex_hull, axis=1)

    # Convert the DataFrame to a GeoDataFrame
    temperature = gpd.GeoDataFrame(temperature, geometry='geometry')
    return temperature


@st.cache_data()
def load_chlorophyll():
    
    chlorophyll = pd.read_pickle('app/data/chlorophyll.pkl')
    min_depth = chlorophyll.depth.min()
    chlorophyll = chlorophyll.query('chlorophyll > 0 and depth == @min_depth')
    return chlorophyll



@st.cache_data()
def load_ruta() -> gpd.GeoDataFrame:
    ruta = (gpd.read_file('app/data/Track_OOO.gpx', layer='tracks')).explode().reset_index(drop=True)
    ruta = ruta.set_crs('epsg:4326')
    ruta['text'] = ruta.name.apply(lambda x: f'ruta {x}')
    return ruta


@st.cache_data()
def load_datos_avistamientos():
    df = pd.read_excel('app/data/datos_avistamientos.xlsx')
    return df


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk
from datetime import datetime


def main():

    st.set_page_config(
    page_title="Avistamiento mamíferos",
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
    


    
    lat_min, lat_max = -41, -39
    # cargamos datos
    df_avistamientos = load_datos_avistamientos()
    ruta = load_ruta()
    chlorophyll = load_chlorophyll()
    temperature = load_temperature(lat_min, lat_max)

    _, col_mapa, _ = st.columns([1, 5, 1])
    dates = [pd.to_datetime(date) for date in df_avistamientos['Fecha'].unique()]
    # #add a slider for selecting dates based on the available dates
    # start_time = st.slider(
    #     "When do you start?",
    #     min_value=dates[0],
    #     max_value=dates[-1],
    #     value=(dates[0], dates[-1]),
    #     format="MM/DD/YY")
    
    with col_mapa:
        especies = df_avistamientos['Especie'].unique()
        for especie in especies:
            st.toggle(f'{especie}', value=True, key=especie)
        #I want a toggle for each species
        species = st.multiselect('Especies', especies, default=especies)
        with st.expander('Datos brutos', expanded=False):
            st.write(df_avistamientos.head())
            st.write(chlorophyll.head())
            st.write(temperature.head())

    
        st.header('Mapa de avistamientos')
        plot_mapa(temperature, ruta, df_avistamientos)




def plot_mapa(temperature, ruta, df_avistamientos):
    color_range = [
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
    # Create bins for temperature
    bins = np.linspace(temperature['temperature'].min(), temperature['temperature'].max(), len(color_range))

    def color_scale(val):
        for i, b in enumerate(bins):
            if val < b:
                return color_range[i]
        return color_range[i]

    temperature["fill_color"] = temperature["temperature"].apply(lambda row: color_scale(row))

    layers = [
        pdk.Layer(
            'PolygonLayer',
            data=temperature.query('time<"2023-01-02"')[['longitude', 'latitude', 'geometry', 'temperature', 'fill_color']],
            get_polygon='geometry.coordinates',
            get_elevation='temperature',
            get_fill_color='fill_color',
            get_line_color=[128, 128, 128], 
            line_width_min_pixels=2,
            opacity=0.5,
            pickable=False,
            stroked=False,
            auto_highlight=True,
            #tooltip={"text": "Temperatura: {temperature} °C"},
        ),
        pdk.Layer(
            'TextLayer',
            data=temperature.query('time<"2023-01-02"')[['longitude', 'latitude', 'geometry', 'temperature_str', 'fill_color']],
            get_position=[ 'longitude', 'latitude'],
            get_text='temperature_str',
            get_color=[128,128,128],
            get_size=10,
            get_alignment_baseline="'bottom'",
        ),
        pdk.Layer(
            'PathLayer',
            data=ruta,
            get_path='geometry.coordinates',
            get_color=[128, 128, 128],  # Changed to gray color
            width_scale=20,
            width_min_pixels=6,
            get_width='width',
            pickable=False,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_avistamientos[['lat', 'lon', 'Especie', 'color']].copy(),
            get_position=['lon', 'lat'],
            get_color='color',  
            get_radius=200,
            pickable=True,
            auto_highlight=True,
            #tooltip=,
        )
    ]

    view_state = pdk.ViewState(
        latitude=-39.5,
        longitude=-73.8,
        zoom=10,
        pitch=50,
    )

    deck = pdk.Deck(
        map_style=pdk.map_styles.LIGHT,
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "Especie: {Especie}"}
    )
    st.pydeck_chart(deck)
    
    
def crop_map(df, lat_min, lat_max, variable='temperature'):
    min_depth = df.depth.min()
    df = df.query(f'latitude >= {lat_min} and latitude <= {lat_max} and depth == @min_depth')
    
    df = df.dropna(how='any')
    return df


def create_grid(temperature, grid_size = 0.08):
    
    temperature['geometry'] = temperature.apply(lambda row: MultiPoint([(row['longitude'] - grid_size / 2, row['latitude'] - grid_size / 2), 
                                                                        (row['longitude'] + grid_size / 2, row['latitude'] - grid_size / 2), 
                                                                        (row['longitude'] + grid_size / 2, row['latitude'] + grid_size / 2), 
                                                                        (row['longitude'] - grid_size / 2, row['latitude'] + grid_size / 2)]).convex_hull, axis=1)
    temperature = gpd.GeoDataFrame(temperature, geometry='geometry')
    return temperature


@st.cache_data()
def load_temperature(lat_min, lat_max):

    temperature = pd.read_parquet('data/temperature.parquet')
    temperature = temperature.pipe(crop_map, lat_min, lat_max).pipe(create_grid, grid_size=0.08)
    temperature['temperature'] = temperature['temperature'].round(1)
    temperature['temperature_str'] = temperature['temperature'].astype(str) + ' °C'
    return temperature


@st.cache_data()
def load_chlorophyll():
    
    chlorophyll = pd.read_parquet('data/chlorophyll.parquet')
    #chlorophyll = chlorophyll.pipe(crop_map, -40, -39, variable='chlorophyll').pipe(create_grid, grid_size=0.2)
    return chlorophyll


@st.cache_data()
def load_ruta() -> gpd.GeoDataFrame:
    ruta = gpd.read_file('data/ruta.geojson')
    return ruta


@st.cache_data()
def load_datos_avistamientos():
    df = pd.read_excel('data/datos_avistamientos.xlsx')
    unique_species = df['Especie'].unique()
    colors = [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)] for _ in range(len(unique_species))]
    species_color = dict(zip(unique_species, colors))
    df['color'] = df['Especie'].map(species_color)
    return df


if __name__ == "__main__":
    main()

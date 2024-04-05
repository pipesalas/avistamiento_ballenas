import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
from streamlit_carousel import carousel



def main():
    
    

    st.set_page_config(
    page_title="Avistamiento mamíferos marinos",
    page_icon="🌊",
    layout="wide",
    menu_items={
        'About': "# Datos de avistamiento marinos realizados por *vuelvealoceano*",
        #'Vuelve Al Oceano': 'https://www.vuelvealoceano.cl',
    }
    )

    lat_min, lat_max = -41, -39
    ruta = load_ruta()
    df_avistamientos, diccionario_color = load_datos_avistamientos()
    chlorophyll = load_chlorophyll(lat_min, lat_max)
    temperature = load_temperature(lat_min, lat_max)

    _, col_mapa, _ = st.columns([1, 5, 1])
   
    
    with col_mapa:
        st.title('🐋 Monitor de mamiferos marinos :ocean:')

        st.markdown('''
                **Bienvenido a nuestra aplicación de avistamiento de observaciones de Ballenas en Chile**, somos [Vuelve al Oceano](http://www.vuelvealoceano.cl) y este
                es nuestro monitor de mamiferos marinos.''')
        st.markdown('''
                    A continuación puedes seleccionar la fecha de avistamiento, la especie y la variable que quieres visualizar.
                    Además, si existen avistamientos en la fecha seleccionada, se mostrarán en el mapa y si hay alguna foto disponible, se mostrará en la sección de fotos.''')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('**Filtro de fechas**')
            dates = [str(date).split(' ')[0] for date in df_avistamientos['Fecha'].unique()]
            start_date = st.selectbox('Fecha de avistamiento', dates)

            st.markdown('**Filtro de variables**')
            variable = st.radio('Seleccionamos una variable', ['Temperatura', 'Clorofila', 'Fitoplancton'], key='variable')
            var = {'Temperatura': 'temperature', 'Clorofila': 'chlorophyll', 'Fitoplancton': 'phyc'}[variable]

        with col2:
            st.markdown('**Filtro de especies**')
            especies_seleccionadas = {}
            #lets order the Especies 

            especies = df_avistamientos['Especie'].unique()
            especies.sort()

            for especie in especies[:int(len(especies)/2+1)]:
                especies_seleccionadas[especie] = st.toggle(especie, value=True, key=especie)

        with col3:
            #st.markdown('<span style="color: red;">text</span> ', unsafe_allow_html=True)
            st.markdown('    ', unsafe_allow_html=True)
            for especie in especies[int(len(especies)/2+1):]:
                especies_seleccionadas[especie] = st.toggle(especie, value=True, key=especie)
        
        filtro_especies = [especie for especie, seleccion in especies_seleccionadas.items() if seleccion]
        df_avistamientos = df_avistamientos.query('Especie in @filtro_especies')


        
            
        with st.expander('Conteo de especies', expanded=True):
            plot_conteo_especies(df_avistamientos)
            conteo_especie_tiempo(df_avistamientos)
            

    
        st.header('Mapa de avistamientos')
        df_mapa = {'temperature': temperature, 'chlorophyll': chlorophyll, 'phyc': chlorophyll}[var]
        plot_mapa(df_mapa.query('time==@start_date'), ruta, df_avistamientos.query('Fecha==@start_date'), var, diccionario_color)
        if len(df_avistamientos.query('Fecha==@start_date')) == 0:
            st.warning('No hay avistamientos en la fecha seleccionada')

    
        ploteamos_fotos(start_date)


        

def get_correct_chilean_date(date):
    date = pd.to_datetime(date)
    chilean_date = ''
    if date.day < 10:
        chilean_date += f'0{date.day}_'
    else:
        chilean_date += f'{date.day}_'
    if date.month < 10:
        chilean_date += f'0{date.month}_{date.year}'
    else:
        chilean_date += f'{date.month}_{date.year}'
    return chilean_date


def ploteamos_fotos(start_date):
    chilean_date = get_correct_chilean_date(start_date)
    files = os.listdir('data/fotos')
    files = [os.path.join('data/fotos', file) for file in files]
    fotos_day = [file for file in files if chilean_date in file]
    if len(fotos_day) == 0:
        st.warning('No hay fotos en la fecha seleccionada')
    else:
        st.header('Fotos de avistamientos')
        test_items = []
        for file in fotos_day:
            #st.write(f"https://github.com/pipesalas/avistamiento_ballenas/blob/main/{file}?raw=true")
            test_items.append(dict(
                                title=f"",
                                text=f"",
                                interval=None,
                                img=f"https://github.com/pipesalas/avistamiento_ballenas/blob/main/{file}?raw=true",
                            ))
        
        carousel(items=test_items, width=1, height=1000)



def conteo_especie_tiempo(df_avistamientos):
    df_number_avistamientos = df_avistamientos.groupby('Fecha').size().reset_index(name='counts')
    fig = px.bar(df_number_avistamientos, 
            x='Fecha', 
            y='counts', 
            labels={'x':'Fecha', 'y':'Numero de avistamientos a lo largo del tiempo'},
                         title='Numero de avistamientos por fecha',
                         width=800, height=400)
    fig.update_yaxes(title_text='Conteo')
    st.plotly_chart(fig)


def plot_conteo_especies(df_avistamientos):
    species_counts = df_avistamientos['Especie'].value_counts()
    fig = px.bar(species_counts, 
                 y=species_counts.values, 
                 x=species_counts.index, 
                 labels={'x':'Species', 'y':'Count'}, 
                 title='Conteo de avistamientos por especie',
                 width=800, height=400)
    #update xaxis name
    fig.update_xaxes(title_text='Especies')
    fig.update_yaxes(title_text='Conteo')
    st.plotly_chart(fig)


def plot_mapa(dataf, ruta, df_avistamientos, variable, diccionario_color):
    color_range = {'temperature': [
        [255, 255, 204],  # Light Yellow
        [255, 237, 160],  # Lighter Yellow
        [254, 217, 118],  # Light Orange
        [254, 178, 76],   # Orange
        [253, 141, 60],   # Darker Orange
        [252, 78, 42],    # Dark Orange
        [227, 26, 28],    # Light Red
        [189, 0, 38],     # Red
    ],
    'chlorophyll': [
        [237, 248, 229],  # Pale Green
        [199, 233, 192],  # Lighter Green
        [161, 217, 155],  # Light Green
        [116, 196, 118],  # Green
        [65, 171, 93],   # Darker Green
        [35, 139, 69],   # Dark Green
        [0, 109, 44],    # Deep Green
        [0, 68, 27],     # Deepest Green
    ],
    'phyc': [
        [237, 248, 229],  # Pale Green
        [199, 233, 192],  # Lighter Green
        [161, 217, 155],  # Light Green
        [116, 196, 118],  # Green
        [65, 171, 93],   # Darker Green
        [35, 139, 69],   # Dark Green
        [0, 109, 44],    # Deep Green
        [0, 68, 27],     # Deepest Green
    ]}[variable]

    bins = np.linspace(dataf[variable].min(), dataf[variable].max(), len(color_range))

    def color_scale(val):
        for i, b in enumerate(bins):
            if val < b:
                return color_range[i]
        return color_range[i]

    dataf["fill_color"] = dataf[variable].apply(lambda row: color_scale(row))

    layers = [
        pdk.Layer(
            'PolygonLayer',
            data=dataf[['longitude', 'latitude', 'geometry', variable, 'fill_color']],
            get_polygon='geometry.coordinates',
            get_elevation=variable,
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
            data=dataf[['longitude', 'latitude', 'geometry', f'{variable}_str', 'fill_color']],
            get_position=['longitude', 'latitude'],
            get_text=f'{variable}_str',
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
            data=df_avistamientos[['longitude', 'latitude', 'Especie', 'color', 'text', 'numero_individuos_log']].copy(),
            get_position=['longitude', 'latitude'],
            get_color='color',  
            get_radius='200*numero_individuos_log',
            pickable=True,
            auto_highlight=True,
            #tooltip=,
        )
    ]

    view_state = pdk.ViewState(
        latitude=-39.92,
        longitude=-73.7,
        zoom=11,
        pitch=50,
    )

    deck = pdk.Deck(
        map_style=pdk.map_styles.LIGHT,
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "text": "{text}"
        }
    )
    st.pydeck_chart(deck)

    
    
    
def crop_map(df, lat_min, lat_max, variable='temperature', agg='mean'):
    min_depth = df.depth.min()
    df = df.query(f'latitude >= {lat_min} and latitude <= {lat_max}')
    if variable == 'temperature':
        temp_surface = df.query(f'depth == {min_depth}').copy().rename(columns={variable: 'surface_temperature'})
        temp_surface['surface_temperature'] = temp_surface['surface_temperature'].round(1)
        df = df.groupby(['time', 'latitude', 'longitude']).agg({variable: agg}).reset_index()
        df = df.merge(temp_surface[['time', 'latitude', 'longitude', 'surface_temperature']], on=['time', 'latitude', 'longitude'], how='left')

    elif variable == 'chlorophyll':
        df = df.groupby(['time', 'latitude', 'longitude']).agg({variable: agg, 'phyc': agg}).reset_index()
        
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
    temperature = temperature.pipe(crop_map, lat_min, lat_max, variable='temperature').pipe(create_grid, grid_size=0.08)
    temperature['temperature'] = temperature['temperature'].round(1)
    temperature['temperature_str'] = temperature['temperature'].astype(str) + ' °C'
    temperature['surface_temperature_str'] = temperature['surface_temperature'].astype(str) + ' °C'
    return temperature


@st.cache_data()
def load_chlorophyll(lat_min, lat_max):
    
    chlorophyll = pd.read_parquet('data/chlorophyll.parquet')
    chlorophyll = chlorophyll.pipe(crop_map, lat_min, lat_max, variable='chlorophyll').pipe(create_grid, grid_size=0.2)
    chlorophyll['chlorophyll'] = chlorophyll['chlorophyll'].round(1)
    chlorophyll['phyc'] = chlorophyll['phyc'].round(1)
    chlorophyll['phyc_str'] = chlorophyll['phyc'].astype(str) + ' mmoles/m^3'
    chlorophyll['chlorophyll_str'] = chlorophyll['chlorophyll'].astype(str) + ' g/m^3'

    return chlorophyll


@st.cache_data()
def load_ruta() -> gpd.GeoDataFrame:
    ruta = gpd.read_file('data/ruta.geojson')
    return ruta


@st.cache_data()
def load_datos_avistamientos():
    df = pd.read_excel('data/datos_avistamientos.xlsx')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    unique_species = df['Especie'].unique()
    colors = [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)] for _ in range(len(unique_species))]
    species_color = dict(zip(unique_species, colors))
    df = df.query('Fecha >= "2023-01-01"')
    df['color'] = df['Especie'].map(species_color)
    df.loc[:, 'numero_individuos_log'] = np.log(df['individuos'] + 1)

    df.loc[:, 'text'] = df.apply(lambda row: f"Especie: {row['Especie']}, \nNumero de individuos: {row['individuos']}, \nObservaciones: {row['Observaciones']}" if pd.notnull(row['Observaciones']) else f"Especie: {row['Especie']}, \nNumero de individuos: {row['individuos']}", axis=1)
    
    return df, species_color


if __name__ == "__main__":
    main()

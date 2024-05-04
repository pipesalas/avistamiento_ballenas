import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString, Polygon
import streamlit as st
import pydeck as pdk

import plotly.express as px
import plotly.graph_objects as go
import os
from streamlit_carousel import carousel
import PIL.Image as Image
from config import zona_1, zona_2, zona_3, zona_4
import textwrap
import gpxpy
import glob
from datetime import datetime

def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append([point.latitude, point.longitude])

    return data




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
    rutas = load_ruta()   
    df_avistamientos, diccionario_color = load_datos_avistamientos()
    df_conteo_directo = load_datos_avistamientos_orilla()    
    
    chlorophyll = load_chlorophyll(lat_min, lat_max)
    temperature = load_temperature(lat_min, lat_max)
    
    _, col_mapa, _ = st.columns([1, 10, 1])
   
    
    with col_mapa:
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            image = Image.open('data/logo_vuelvealoceano.png')
            st.image(image, use_column_width=True,)
        col_title.title('Monitoreo de mamíferos marinos en las localidades de Huiro y Chaihuin')
        col_title.caption('Proyecto financiado por TNC Chile y GORE Los Ríos ')

        st.markdown('''El objetivo de este proyecto es conocer qué especies de a mamíferos marinos transitan por el área de estudio (Huiro y Chaihuin, comuna de Corral, Los Ríos, Chile) y qué
                     comportamientos tienen en la zona. ¿Acaso se alimentan?, ¿descansan?, ¿se reproducen?. Son preguntas que intentamos responder con este monitoreo, con el objetivo de recopilar información y
                     proponer medidas de protección para estos animales.

En esta aplicación podrás ver las observaciones que se han realizado durante el proyecto de investigación que llevamos realizando. Algunas de las observaciones se han realizado durante
navegaciones de monitoreo con un equipo de voluntari@s, y otras observaciones han realizado desde tierra vecinas y vecinos de las localidades a través de un chat de whatsapp
que creamos con este fin. Agradecemos a cada persona que observa el mar y comparte sus avistamientos. 

A continuación puedes seleccionar la fecha y el factor ambiental que quieras visualizar. Además, si existen fotos de ese día se mostrarán en la sección de fotos.''')
        _, col, _ = st.columns([1, 10, 1])    
        with col:
            tab1, tab2 = st.tabs(['Conteo de especies', 'Conteo por fecha'])
            with tab1:
                plot_conteo_especies(df_avistamientos, df_conteo_directo)
            with tab2:
                conteo_especie_tiempo(df_avistamientos, df_conteo_directo)
            

    
        st.header('Mapas de avistamientos')
        tab_barco, tab_orilla = st.tabs(['Avistamientos desde barco', 'Avistamientos desde orilla'])
        with tab_barco:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown('**Filtro de fechas**')
                total_dates = list(df_avistamientos['Fecha'].unique())
                total_dates.sort()
                total_dates = [pd.to_datetime(date).strftime('%d-%m-%Y') for date in total_dates]
                date_selected = st.selectbox('Selecciona una fecha de avistamiento', total_dates, index=0, key='fecha_avistamiento_barco')
                start_date = pd.to_datetime(date_selected, format="%d-%m-%Y").strftime('%Y-%m-%d')

                st.markdown('**Filtro de variables**')
                variable = st.radio('Coloreamos con una variable', ['Temperatura', 'Clorofila', 'Fitoplancton'], key='variable')
                var = {'Temperatura': 'temperature', 'Clorofila': 'chlorophyll', 'Fitoplancton': 'phyc'}[variable]

            with col2:
                df_mapa = {'temperature': temperature, 'chlorophyll': chlorophyll, 'phyc': chlorophyll}[var]
                if len(df_avistamientos.query('Fecha==@start_date')) == 0:
                    st.warning('No hay avistamientos en la fecha seleccionada')
                else:
                    plot_mapa(df_mapa.query('time==@start_date'), rutas.query('date==@start_date'), df_avistamientos.query('Fecha==@start_date'), var, diccionario_color)


        with tab_orilla:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown('**Filtro de fechas**')
                total_dates = list(df_conteo_directo['Fecha'].unique())
                total_dates.sort()
                total_dates = [pd.to_datetime(date).strftime('%d-%m-%Y') for date in total_dates]
                date_selected = st.selectbox('Selecciona una fecha de avistamiento', total_dates, index=0, key='fecha_avistamiento_orilla')
                start_date = pd.to_datetime(date_selected, format="%d-%m-%Y").strftime('%Y-%m-%d')
        
            with col2:
                if len(df_conteo_directo.query('Fecha==@start_date')) == 0:
                    st.warning('No hay avistamientos en la fecha seleccionada')
                else:
                    plot_conteo_directo(df_conteo_directo, start_date)


    
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
            test_items.append(dict(
                                title=f"",
                                text=f"",
                                interval=None,
                                img=f"https://github.com/pipesalas/avistamiento_ballenas/blob/main/{file}?raw=true",
                            ))
        
        carousel(items=test_items, width=1, height=2000)



def conteo_especie_tiempo(df_avistamientos,  df_conteo_directo, width=800, height=400):
    df_toplot = pd.concat((df_avistamientos[['Fecha', 'Especie']].copy(), df_conteo_directo[['Fecha', 'Especie']].copy()), axis=0)
    df_number_avistamientos = df_toplot.groupby('Fecha').size().reset_index(name='counts')
    fig = px.bar(df_number_avistamientos, 
            x='Fecha', 
            y='counts', 
            labels={'x':'Fecha', 'y':'Número de avistamientos a lo largo del tiempo'},
                         title='Número de avistamientos por fecha',
                         width=width, height=height)
    fig.update_yaxes(title_text='Conteo')
    st.plotly_chart(fig)
    

def plot_conteo_especies(df_avistamientos, df_conteo_directo, width=800, height=400):
    
    df_toplot = pd.concat((df_avistamientos['Especie'].copy(), df_conteo_directo['Especie'].copy()), axis=0)
    df_toplot = df_toplot.apply(lambda x: x.split(' (')[0])
    species_counts = df_toplot.value_counts()
    fig = px.bar(species_counts, 
                 #y=species_counts.values[0], 
                 #x=species_counts.index, 
                 labels={'x':'Species', 'y':'Count'}, 
                 title='Número de avistamientos por especie',
                 width=width, height=height)
    fig.update_layout(showlegend=False)
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
        longitude=-73.65,
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


def plot_conteo_directo(df_conteo_directo, fecha, width=800, height=400):
    
    textos = df_conteo_directo.groupby(['Fecha'])['text'].apply(lambda x: ', '.join(x)).reset_index()
    textos['text'] = textos['text'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)))
    df_conteo_directo = (df_conteo_directo
                         .query('Fecha == @fecha')
                         .merge(textos.rename({'text': 'texto_completo'}, axis=1), on='Fecha', how='left'))
    df_toplot = df_conteo_directo.drop_duplicates(subset=['Fecha'])
    
    layers = [
        pdk.Layer(
            'PolygonLayer',
            data=df_toplot[['geometry', 'color', 'Numero', 'texto_completo']],
            get_polygon='geometry.coordinates',
            #get_elevation='Numero',
            get_fill_color='color',
            #get_line_color=[128, 128, 128], 
            #line_width_min_pixels=2,
            opacity=0.5,
            pickable=True,
            #stroked=False,
            #auto_highlight=True,
            
        ),
    ]

    view_state = pdk.ViewState(
        latitude=-39.93,
        longitude=-73.65,
        zoom=11,
        pitch=50,
    )

    deck = pdk.Deck(
        map_style=pdk.map_styles.LIGHT,
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "text": "{texto_completo}"
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
    
    folder_path = './data/tracks/'
    gpx_files = glob.glob(folder_path + '/*.gpx')
    ruta = merge_gdfs(gpx_files)
    ruta['text'] = ruta['date'].apply(lambda x: f'Fecha: {x}')
    return ruta


def create_geodataframe(date, coordinates):
    new_coordinates = [(coord[1], coord[0]) for coord in coordinates]
    
    line = [LineString(new_coordinates)]
    gdf = gpd.GeoDataFrame(geometry=line)
    gdf = gdf.assign(date=date)

    date_format = '%d/%m/%Y'
    gdf['date'] = gdf['date'].apply(pd.to_datetime,)# format=date_format)
    
    return gdf

def merge_gdfs(gpx_files):
    gdfs = []
    for file in gpx_files:
        date = file.split('/')[-1].split('.')[0].replace(':','/')
        year = date.split('/')[2]
        day = date.split('/')[0]
        month = date.split('/')[1]
        gdfs.append(create_geodataframe(date=f'{year}/{month}/{day}', coordinates=parse_gpx(file)))

    return pd.concat(gdfs)

@st.cache_data()
def load_datos_avistamientos():
    df = pd.read_excel('data/datos_avistamientos.xlsx')
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
    unique_species = df['Especie'].unique()
    colors = [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)] for _ in range(len(unique_species))]
    species_color = dict(zip(unique_species, colors))
    df = df.query('Fecha >= "2023-01-01"')
    df['color'] = df['Especie'].map(species_color)
    df.loc[:, 'numero_individuos_log'] = np.log(df['individuos'] + 1)

    df['Observaciones'] = df['Observaciones'].apply(lambda x: 'Observaciones: ' + x if pd.notnull(x) else x)
    df['Observaciones'] = df['Observaciones'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)) if pd.notnull(x) else x)
    df.loc[:, 'text'] = df.apply(lambda row: f"Especie: {row['Especie']} \nNumero de individuos: {row['individuos']} \n{row['Observaciones']}", axis=1)
    
    return df, species_color


@st.cache_data()
def load_datos_avistamientos_orilla():
    df = pd.read_excel('data/Planilla conteo directo .xlsx', skiprows=1)
    #df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    zonas = {1: Polygon(zona_1['coordinates'][0]), 2: Polygon(zona_2['coordinates'][0]), 3: Polygon(zona_3['coordinates'][0]), 4: Polygon(zona_4['coordinates'][0])}
    df.columns = [col.strip() for col in df.columns]
    df['color'] = df['Subzona'].map({
                                    1: [255, 255, 0], 
                                    2: [255, 0, 0], 
                                    3: [0, 0, 255], 
                                    4: [0, 128, 0]})
    
    df['geometry'] = df['Subzona'].map(zonas)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df['Especie_short'] = df['Especie'].apply(lambda x: x.split(' (')[0])
    df['text'] = df.apply(lambda row: f"- {row['Especie_short']}: ({row['Numero']} ejemplares)  {row['Observaciones']}", axis=1)

    return df





if __name__ == "__main__":
    main()

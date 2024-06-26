import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString, Polygon
import pydeck as pdk

import plotly.express as px
import plotly.graph_objects as go
import os
import streamlit as st
from streamlit_carousel import carousel
import PIL.Image as Image
from config import zona_1, zona_2, zona_3, zona_4
import textwrap
import gpxpy
import glob
from datetime import datetime
from typing import Union, List




def main():

    st.set_page_config(
    page_title="Avistamiento mamíferos marinos",
    page_icon="🌊",
    layout="wide",
    menu_items={
        'About': "Datos de avistamiento marinos realizados por *vuelve al oceano* https://www.vuelvealoceano.cl",
    }
    )

    lat_min, lat_max = -41, -39
    with st.spinner('Leyendo rutas de GPS...'):
        rutas = load_ruta()   
    with st.spinner('Leyendo datos de avistamientos...'):
        df_avistamientos, diccionario_color = load_datos_avistamientos()
        df_conteo_directo = load_datos_avistamientos_orilla()    
    with st.spinner('Leyendo datos satelitales de clorofila...'):
        chlorophyll = load_chlorophyll(lat_min, lat_max)
    with st.spinner('Leyendo datos satelitales de temperatura...'):
        temperature = load_temperature(lat_min, lat_max)
    
    _, col_mapa, _ = st.columns([1, 10, 1])
    
    with col_mapa:
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            plot_logo()

        col_title.title('Monitoreo de mamíferos marinos en las localidades de Huiro y Chaihuin')
        col_title.caption('Proyecto financiado por TNC Chile, GORE Los Ríos y ONG Vuelve Al Océano.')

        st.markdown('''El objetivo de este proyecto es conocer qué especies de mamíferos marinos transitan por el área de estudio (Huiro y Chaihuin, comuna de Corral, Los Ríos, Chile) 
                    y qué comportamientos tienen en la zona. ¿Acaso se alimentan?, ¿descansan?, ¿se reproducen?. Respecto de saber sobre sus dinámicas poblacionales, ¿cuántos individuos son?, 
                    ¿regresan todos los años?, ¿cuándo están y cuánto tiempo permanecen en el área?. Son preguntas que intentamos responder con los datos obtenidos en este monitoreo,
                     con el objetivo de entender este ecosistema marino y proponer medidas de conservación para estas especies de mamíferos marinos, que en su mayoría, están en peligro de extinción. 


En esta aplicación podrás ver los avistamientos que se han realizado durante el proyecto de investigación que llevamos realizando desde el año 2022 hasta la fecha. Algunas de las observaciones 
se han realizado durante navegaciones de monitoreo con nuestro equipo de profesionales y voluntari@s previamente capacitados sobre la metodología de estudio. Los avistamientos desde tierra, 
reportados por vecinos y vecinas de las localidades, fueron comunicados a través de un chat de WhatsApp que fue creado para este fin. Agradecemos a cada persona que observa el mar y comparte sus avistamientos.
''')
        _, col, _ = st.columns([1, 10, 1])    
        with col:
            tab1, tab2 = st.tabs(['Conteo de especies', 'Conteo por fecha'])
            with tab1:
                plot_conteo_especies(df_avistamientos, df_conteo_directo)
            with tab2:
                conteo_especie_tiempo(df_avistamientos, df_conteo_directo)
            

    
        st.header('Mapas de avistamientos')
        st.markdown('''A continuación puedes seleccionar la fecha y el factor ambiental que quieras visualizar. Además, si existen fotos de ese día se mostrarán en la sección de fotos.''')
        tab_barco, tab_orilla = st.tabs(['Avistamientos desde barco', 'Avistamientos desde orilla'])
        with tab_barco:
            col1, col2 = st.columns([1, 4])
            with col1:
                start_date_barco = filtro_fechas(df_avistamientos, todas_las_fechas=True)

                st.markdown('**Filtro de variables**')
                variable = st.radio('Coloreamos con una variable', ['Temperatura', 'Clorofila', 'Fitoplancton'], key='variable')
                var = {'Temperatura': 'temperature', 'Clorofila': 'chlorophyll', 'Fitoplancton': 'phyc'}[variable]
                df_mapa = {'temperature': temperature, 'chlorophyll': chlorophyll, 'phyc': chlorophyll}[var]

            with col2:
                if len(df_avistamientos.query('Fecha==@start_date_barco')) == 0 and start_date_barco != "Todas las fechas":
                    st.warning('No hay avistamientos en la fecha seleccionada')
                elif start_date_barco == "Todas las fechas":
                    plot_mapa(df_mapa.query('time=="2025-01-01"'), rutas.query('date=="2023-03-22"'), df_avistamientos, var)
                else:
                    plot_mapa(df_mapa.query('time==@start_date_barco'), rutas.query('date==@start_date_barco'), df_avistamientos.query('Fecha==@start_date_barco'), var)

            ploteamos_fotos(start_date_barco)


        with tab_orilla:
            col1, col2 = st.columns([1, 4])
            with col1:
                start_date_orilla = filtro_fechas(df_conteo_directo, todas_las_fechas=True, key='avistamiento_orilla')
        
            with col2:
                if len(df_conteo_directo.query('Fecha==@start_date_orilla')) == 0 and start_date_orilla != "Todas las fechas":
                    st.warning('No hay avistamientos en la fecha seleccionada')
                else:
                    plot_conteo_directo(df_conteo_directo, start_date_orilla)

            ploteamos_fotos(start_date_orilla, folder='data/fotos_conteo_directo', sep=':')

        plot_comportamiento(df_avistamientos)



def plot_comportamiento(df_avistamientos: pd.DataFrame) -> None:
    """
    Plots the behavior of marine mammals based on the given data.

    Args:
        df (pd.DataFrame): The DataFrame containing the behavior data.

    Returns:
        None
    """
    df_toplot = (df_avistamientos.groupby(['Especie', 'Comportamiento'], as_index=False)['Fecha']
                                 .count()
                                 .rename(columns={'Fecha':'Conteo'})
                                 .sort_values(by='Conteo', ascending=False)
                                 .copy())
    fig = px.bar(df_toplot, 
                 x='Especie', 
                 y='Conteo', 
                 color='Comportamiento', 
                 title='Comportamiento de las especies', 
                 barmode='stack', 
                 width=1000, 
                 height=600, 
                 text_auto=True)

    st.plotly_chart(fig)




def plot_logo() -> None:
    """
    Displays the vuelvealoceano logo image on the Streamlit app.
    """
    image = Image.open('data/logo_vuelvealoceano.png')
    st.image(image, use_column_width=True)


def filtro_fechas(df_avistamientos: pd.DataFrame, 
                  todas_las_fechas: bool = False, 
                  key: str = 'avistamiento_barco') -> datetime:
    """
    Filter dates based on user selection.

    Args:
        df_avistamientos (pd.DataFrame): The DataFrame containing the avistamientos data.
        todas_las_fechas (bool, optional): Whether to include all dates in the selection. Defaults to False.
        key (str, optional): The key used for caching the selection. Defaults to 'avistamiento_barco'.

    Returns:
        datetime: The selected start date in the format '%Y-%m-%d'.
    """

    st.markdown('**Filtro de fechas**')
    total_dates = list(df_avistamientos['Fecha'].unique())
    total_dates.sort()
    total_dates = [pd.to_datetime(date).strftime('%d-%m-%Y') for date in total_dates]
    if todas_las_fechas:
        total_dates = ['Todas las fechas'] + total_dates
    date_selected = st.selectbox('Selecciona una fecha de avistamiento', total_dates, index=0, key=key)
    if date_selected == 'Todas las fechas':
        start_date = "Todas las fechas"
    else:
        start_date = pd.to_datetime(date_selected, format="%d-%m-%Y").strftime('%Y-%m-%d')
    return start_date



def get_correct_chilean_date(date: Union[str, pd.Timestamp], 
                             sep: str = '_') -> str:
    """
    Converts a given date to the correct Chilean date format.

    Args:
        date (str or pd.Timestamp): The date to be converted.
        sep (str, optional): The separator to be used in the Chilean date format. Defaults to '-'.

    Returns:
        str: The converted Chilean date in the format 'DD_MM_YYYY'.

    Example:
        >>> get_correct_chilean_date('2022-01-15')
        '15_01_2022'
    """
    date = pd.to_datetime(date)
    chilean_date = ''
    if date.day < 10:
        chilean_date += f'0{date.day}{sep}'
    else:
        chilean_date += f'{date.day}{sep}'
    if date.month < 10:
        chilean_date += f'0{date.month}{sep}{date.year}'
    else:
        chilean_date += f'{date.month}{sep}{date.year}'
    return chilean_date


def ploteamos_fotos(start_date: Union[str, pd.Timestamp], 
                    sep: str = '_', 
                    folder: str = 'data/fotos'):
    """
    Plots photos of whale sightings based on the given start date.

    Args:
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        sep (str, optional): The separator used in the filenames. Defaults to '_'.

    Returns:
        None
    """
    files = os.listdir(folder)
    files = [os.path.join(folder, file) for file in files]
    if start_date == 'Todas las fechas':
        fotos_day = files
    else:
        chilean_date = get_correct_chilean_date(start_date, sep=sep)
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
        
        carousel(items=test_items, width=1, height=1500)



def conteo_especie_tiempo(df_avistamientos: pd.DataFrame, 
                          df_conteo_directo: pd.DataFrame, 
                          width: int = 800, 
                          height: int = 400) -> None:
    """
    Plots the number of sightings over time for different species.
    
    Args:
        df_avistamientos (pd.DataFrame): DataFrame containing sighting data with columns 'Fecha' and 'Especie'.
        df_conteo_directo (pd.DataFrame): DataFrame containing direct count data with columns 'Fecha' and 'Especie'.
        width (int, optional): Width of the plot. Defaults to 800.
        height (int, optional): Height of the plot. Defaults to 400.
    
    Returns:
        None
    """
    
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
    

def plot_conteo_especies(df_avistamientos: pd.DataFrame, 
                         df_conteo_directo: pd.DataFrame, 
                         width: int = 800, 
                         height: int = 400):
    """
    Plots the count of species sightings.

    Args:
        df_avistamientos (pd.DataFrame): DataFrame containing species sightings data.
        df_conteo_directo (pd.DataFrame): DataFrame containing direct count data.
        width (int, optional): Width of the plot. Defaults to 800.
        height (int, optional): Height of the plot. Defaults to 400.
    """
    
    df_toplot = pd.concat((df_avistamientos['Especie'].copy(), df_conteo_directo['Especie'].copy()), axis=0)
    df_toplot = df_toplot.apply(lambda x: x.split(' (')[0])
    species_counts = df_toplot.value_counts()
    fig = px.bar(species_counts, 
                 labels={'x':'Species', 'y':'Count'}, 
                 title='Número de avistamientos por especie',
                 width=width, height=height)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text='Especies')
    fig.update_yaxes(title_text='Conteo')
    st.plotly_chart(fig)


def plot_mapa(dataf: pd.DataFrame, 
              ruta: gpd.GeoDataFrame, 
              df_avistamientos: pd.DataFrame, 
              variable: str) -> None:
    """
    Plots a map with various layers using the given dataframes and variable.

    Args:
        dataf (pd.DataFrame): The dataframe containing the main data for the map.
        ruta (gpd.GeoDataFrame): The geodataframe containing the route data for the map.
        df_avistamientos (pd.DataFrame): The dataframe containing the avistamientos data for the map.
        variable (str): The variable to be used for coloring the map.

    Returns:
        None
    """

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

    def color_scale(val: float) -> List[int]:
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
    """
    Plots the count of direct sightings on a map.

    Args:
        df_conteo_directo (DataFrame): The dataframe containing the count of direct sightings.
        fecha (str): The date for which the sightings should be plotted. Use 'Todas las fechas' to plot all dates.
        width (int, optional): The width of the plot in pixels. Defaults to 800.
        height (int, optional): The height of the plot in pixels. Defaults to 400.
    """
    
    df_original = df_conteo_directo.copy()
    textos = df_conteo_directo.groupby(['Fecha'])['text'].apply(lambda x: ', '.join(x)).reset_index()
    textos['text'] = textos['text'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)))
    if fecha == 'Todas las fechas':
        df_toplot = (df_conteo_directo.groupby(['Especie', 'Subzona'])['Numero'].sum().reset_index())                            
        df_toplot['texto_completo'] = df_toplot.apply(lambda row: f"{row['Especie'].split(' (')[0]} ({row['Numero']} ejemplares)", axis=1)

        df_toplot = df_toplot.groupby(['Subzona'])['texto_completo'].apply(lambda x: '\n'.join(x)).reset_index()
        diccionario_zonas = df_original[['Subzona', 'geometry']].drop_duplicates(subset=['Subzona']).set_index('Subzona').to_dict(orient='index')
        diccionario_color = df_original[['Subzona', 'color']].drop_duplicates(subset=['Subzona']).set_index('Subzona').to_dict(orient='index')
        df_toplot['geometry'] = df_toplot['Subzona'].apply(lambda x: diccionario_zonas[x]['geometry'])
        df_toplot['color'] = df_toplot['Subzona'].apply(lambda x: diccionario_color[x]['color'])
        df_toplot = gpd.GeoDataFrame(df_toplot, geometry='geometry')

    else:
        df_conteo_directo = (df_conteo_directo
                         .query('Fecha == @fecha')
                         .merge(textos.rename({'text': 'texto_completo'}, axis=1), on='Fecha', how='left'))
    
        df_toplot = df_conteo_directo.drop_duplicates(subset=['Fecha'])
    layers = [
        pdk.Layer(
            'PolygonLayer',
            data=df_toplot[['geometry', 'color', 'texto_completo']],
            get_polygon='geometry.coordinates',
            get_fill_color='color',
            opacity=0.5,
            pickable=True,
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

    
    
    
def crop_map(df: pd.DataFrame, lat_min: float, lat_max: float, variable: str = 'temperature', agg: str = 'mean') -> pd.DataFrame:
    """
    Crop the map data based on latitude range and perform aggregation on the specified variable.

    Parameters:
    - df (DataFrame): The input DataFrame containing map data.
    - lat_min (float): The minimum latitude value for cropping.
    - lat_max (float): The maximum latitude value for cropping.
    - variable (str, optional): The variable to be aggregated. Defaults to 'temperature'.
    - agg (str, optional): The aggregation method to be used. Defaults to 'mean'.

    Returns:
    - df (DataFrame): The cropped and aggregated DataFrame.

    """
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


def create_grid(temperature: pd.DataFrame, grid_size: float = 0.08) -> gpd.GeoDataFrame:
    """
    Create a grid based on temperature data.

    Parameters:
    - temperature (DataFrame): A DataFrame containing temperature data.
    - grid_size (float, optional): The size of each grid cell. Defaults to 0.08.

    Returns:
    - GeoDataFrame: A GeoDataFrame with the temperature data and the created grid.

    """
    temperature['geometry'] = temperature.apply(lambda row: MultiPoint([(row['longitude'] - grid_size / 2, row['latitude'] - grid_size / 2), 
                                                                        (row['longitude'] + grid_size / 2, row['latitude'] - grid_size / 2), 
                                                                        (row['longitude'] + grid_size / 2, row['latitude'] + grid_size / 2), 
                                                                        (row['longitude'] - grid_size / 2, row['latitude'] + grid_size / 2)]).convex_hull, axis=1)
    temperature = gpd.GeoDataFrame(temperature, geometry='geometry')
    return temperature


@st.cache_data()
def load_temperature(lat_min: float, lat_max: float) -> pd.DataFrame:
    """
    Load temperature data and perform cropping and grid creation.

    Args:
        lat_min (float): The minimum latitude value for cropping.
        lat_max (float): The maximum latitude value for cropping.

    Returns:
        pd.DataFrame: The processed temperature data.

    """
    temperature = pd.read_parquet('data/temperature.parquet')
    temperature = temperature.pipe(crop_map, lat_min, lat_max, variable='temperature').pipe(create_grid, grid_size=0.08)
    temperature['temperature'] = temperature['temperature'].round(1)
    temperature['temperature_str'] = temperature['temperature'].astype(str) + ' °C'
    temperature['surface_temperature_str'] = temperature['surface_temperature'].astype(str) + ' °C'
    return temperature


@st.cache_data()
def load_chlorophyll(lat_min: float, lat_max: float) -> pd.DataFrame:
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


def create_geodataframe(date: str, coordinates: List[List[float]]) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a given date and coordinates.

    Args:
        date (str): The date in string format.
        coordinates (List[List[float]]): List of coordinates in the format [[lat1, lon1], [lat2, lon2], ...].

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the line geometry and the date column.
    """
    new_coordinates = [(coord[1], coord[0]) for coord in coordinates]
    
    line = [LineString(new_coordinates)]
    gdf = gpd.GeoDataFrame(geometry=line)
    gdf = gdf.assign(date=date)

    date_format = '%d/%m/%Y'
    gdf['date'] = gdf['date'].apply(pd.to_datetime,)# format=date_format)
    
    return gdf


def merge_gdfs(gpx_files: List[str]) -> gpd.GeoDataFrame:
    """
    Merge multiple GeoDataFrames created from GPX files into a single GeoDataFrame.

    Args:
        gpx_files (List[str]): A list of file paths to GPX files.

    Returns:
        gpd.GeoDataFrame: A merged GeoDataFrame containing the data from all the GPX files.

    """
    gdfs = []
    for file in gpx_files:
        date = file.split('/')[-1].split('.')[0].replace(':','/')
        year = date.split('/')[2]
        day = date.split('/')[0]
        month = date.split('/')[1]
        gdfs.append(create_geodataframe(date=f'{year}/{month}/{day}', coordinates=parse_gpx(file)))

    return gpd.GeoDataFrame(pd.concat(gdfs))


def create_geodataframe(date: str, coordinates: List[List[float]]) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a given date and coordinates.
    
    Args:
        date (str): The date in string format (e.g., 'dd/mm/yyyy').
        coordinates (List[List[float]]): A list of coordinates in the format [[lat1, lon1], [lat2, lon2], ...].
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing a LineString geometry and the given date.
    """
    new_coordinates = [(coord[1], coord[0]) for coord in coordinates]
    
    line = [LineString(new_coordinates)]
    gdf = gpd.GeoDataFrame(geometry=line)
    gdf = gdf.assign(date=date)
    date_format = '%d/%m/%Y'
    gdf['date'] = gdf['date'].apply(pd.to_datetime,)# format=date_format)
    
    return gdf

def parse_gpx(file_path: str) -> List[List[float]]:
    """
    Parses a GPX file and extracts latitude and longitude data from it.

    Args:
        file_path (str): The path to the GPX file.

    Returns:
        List[List[float]]: A list of lists containing latitude and longitude data.
    """
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append([point.latitude, point.longitude])
    return data





@st.cache_data()
def load_datos_avistamientos():
    """
    Load and process avistamientos data.

    Reads the avistamientos data from an Excel file, performs data cleaning and preprocessing,
    and returns the processed DataFrame along with a dictionary mapping species to colors.

    Returns:
        df (pandas.DataFrame): Processed avistamientos data.
        species_color (dict): Dictionary mapping species to colors.
    """
    df = pd.read_excel('data/datos_avistamientos.xlsx')
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
    unique_species = df['Especie'].unique()
    colors = [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)] for _ in range(len(unique_species))]
    species_color = dict(zip(unique_species, colors))
    df = df.query('Fecha >= "2023-01-01"')
    df['color'] = df['Especie'].map(species_color)
    df.loc[:, 'numero_individuos_log'] = np.log(df['individuos'] + 1)

    df['Observaciones'] = df['Observaciones'].apply(lambda x: 'Observaciones: ' + x if pd.notnull(x) else '')
    df['Observaciones'] = df['Observaciones'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)) if pd.notnull(x) else '')
    df.loc[:, 'text'] = df.apply(lambda row: f"Especie: {row['Especie']} \nNumero de individuos: {row['individuos']} \n{row['Observaciones']}", axis=1)
    
    return df, species_color


@st.cache_data()
def load_datos_avistamientos_orilla():
    """
    Loads avistamientos orilla data from an Excel file and performs data processing.

    Returns:
        df (GeoDataFrame): Processed avistamientos orilla data.
    """
    df = pd.read_excel('data/Planilla conteo directo .xlsx', skiprows=1)
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

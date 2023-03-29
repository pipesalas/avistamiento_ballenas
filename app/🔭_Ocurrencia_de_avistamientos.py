import pandas as pd
import numpy as np
import geopandas as gpd
from config import DATES
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk
import os
import requests


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

    st.title(':ocean: Visualizador de las condiciones marítimas  :whale: 🐬')

    #st.('''Para visualizar las condiciones maritimas usamos google earth engine, sacamos la MEDIANA del mes, blabla''' )
    st.markdown('''*¡Bienvenido a nuestra aplicación Ocurrencias de Ballenas en Chile! *''')
    st.markdown('''Esta aplicación le permite explorar las ocurrencias de especies de ballenas registradas en el país de Chile, utilizando datos de Global Biodiversity Information Facility (GBIF). 
    Con esta aplicación, puede ver una tabla de datos de ocurrencia para cada especie de ballena, así como un mapa de las ubicaciones donde se han observado estas especies. 
    Puede acercar y alejar el mapa y hacer clic en marcadores individuales para ver más información sobre cada ocurrencia. 
    Esta aplicación es una gran herramienta para científicos, educadores y cualquier persona interesada en aprender más sobre la distribución de las especies de ballenas en Chile.''')

    
    diccionario_meses = {'septiembre 2022': '2022-09-01_to_2022-10-01',
                        'octubre 2022': '2022-10-01_to_2022-11-01',
                        'noviembre 2022': '2022-11-01_to_2022-12-01',
                        'diciembre 2022': '2022-12-01_to_2023-01-01',
                        'enero 2023': '2023-01-01_to_2023-02-01',
                        'febrero 2023': '2023-02-01_to_2023-03-01',}
    table_columns = ['gbifID', 'species', 'decimalLatitude', 'decimalLongitude', 'eventDate']
    whale_species = {
        'Blue whale': 2440735,
        'Fin whale': 2440718,
        'Humpback whale': 5220086,
        'Killer whale': 2440483,
        'Gray whale': 2440704
    }

    col1, col2 = st.columns(2)
    mes_seleccionado = col1.selectbox('Seleccionamos un mes', diccionario_meses.keys())
    whale_species_selection = col2.selectbox('Select whale species', list(whale_species.keys()))
        

   

    # cargamos datos
    gdf_temp = load_temperature_geodataframe(diccionario_meses.get(mes_seleccionado))
    gdf_chlor = load_chlorophyll_geodataframe(diccionario_meses.get(mes_seleccionado))
    gdf_ballenas = request_gbif_api(whale_species.get(whale_species_selection))
    ruta = load_ruta()

    col1, col2 = st.columns(2)

    with col1:
        st.header('Clorofila')
        plot_chlorophyll(gdf_chlor, gdf_ballenas)
        st.caption('''Se sabe que las ballenas se alimentan de fitoplancton, que son plantas microscópicas que viven en el océano. 
        Estas plantas dependen de la luz solar y los nutrientes para crecer y, como resultado, su abundancia a menudo está relacionada con la concentración de clorofila en el agua. 
        La clorofila es un pigmento verde que ayuda a estas plantas a convertir la luz solar en energía a través de la fotosíntesis. 
        Por lo tanto, las altas concentraciones de clorofila en el agua suelen ser un indicador de la gran abundancia de fitoplancton, que a su vez puede atraer a las ballenas a la zona. 
        En otras palabras, las áreas con altas concentraciones de clorofila pueden ser buenos lugares de alimentación para las ballenas, y monitorear los niveles de clorofila puede ayudarnos
        a comprender mejor el comportamiento y la distribución de las ballenas.''')
        
    with col2:
        st.header('Temperatura')
        plot_temperature(gdf_temp, gdf_ballenas)
        st.caption('''La relación entre la presencia de fitoplancton y la temperatura del mar es compleja y, a menudo, depende de una variedad de factores. 
        Generalmente, el fitoplancton prospera en aguas más cálidas, pero hay muchos otros factores que entran en juego, como la disponibilidad de nutrientes, los niveles de luz y el movimiento del agua. 
        La temperatura ciertamente puede desempeñar un papel en el crecimiento y la distribución del fitoplancton, pero no es el único factor determinante.
        Del mismo modo, la relación entre la temperatura y la presencia de ballenas también es compleja. Algunas especies de ballenas prefieren aguas más frías, mientras que otras prefieren aguas más cálidas. 
        Además, el comportamiento de las ballenas puede verse influenciado por factores como la disponibilidad de presas, los patrones de apareamiento y las rutas de migración. 
        Si bien la temperatura ciertamente puede ser un factor para determinar el comportamiento de las ballenas, es solo uno de los muchos factores que deben tenerse en cuenta.''')


    



def plot_chlorophyll(gdf, gdf_ballenas):

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

    ruta = load_ruta()

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf,
        opacity=0.2,
        get_polygon="geometry.coordinates",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=False,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
    )


    ruta = pdk.Layer(
                    'PathLayer',
                    data=ruta,
                    get_path='geometry.coordinates',
                    get_color=[255, 255, 0],
                    width_scale=20,
                    width_min_pixels=2,
                    get_width='width',
                    pickable=True,
                    )

    layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=gdf_ballenas,
                    get_position=['decimalLongitude', 'decimalLatitude'],
                    get_radius=500,
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    auto_highlight=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    wireframe=True,
                    get_line_color=[255, 255, 255],
                    get_line_width=1,
                    )

    
    ICON_URL = "https://em-content.zobj.net/thumbs/240/facebook/327/whale_1f40b.png"

    icon_data = {
        "url": ICON_URL,
        "width": 2400,
        "height": 2400,
        "anchorY": 2400,
    }

    data = gdf_ballenas
    data["icon_data"] = None
    for i in data.index:
        data["icon_data"][i] = icon_data



    icon_layer = pdk.Layer(
        type="IconLayer",
        data=data,
        get_icon="icon_data",
        get_size=4,
        size_scale=15,
        get_position=['decimalLongitude', 'decimalLatitude'],
        pickable=True,
    )

    r = pdk.Deck(
        [polygon_layer, ruta, layer, icon_layer],
        initial_view_state=define_viewstate(),
        map_style=pdk.map_styles.LIGHT,
        tooltip={'html': "{text}"},
    )
    st.pydeck_chart(r)


def plot_temperature(gdf_temp, gdf_ballenas):

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

    ruta = load_ruta()

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_temp,
        opacity=0.2,
        stroked=False,
        get_polygon="geometry.coordinates",
        filled=True,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        pickable=True,
    )
   
    ruta = pdk.Layer(
        'PathLayer',
        data=ruta,
        get_path='geometry.coordinates',
        get_color=[255, 255, 0],
        width_scale=20,
        width_min_pixels=2,
        get_width='width',
        pickable=False,
        )

    layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=gdf_ballenas,
                    get_position=['decimalLongitude', 'decimalLatitude'],
                    get_radius=500,
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    auto_highlight=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    wireframe=True,
                    get_line_color=[255, 255, 255],
                    get_line_width=1,
                    )

    r = pdk.Deck(
        [polygon_layer, layer, ruta],
        initial_view_state=define_viewstate(),
        map_style=pdk.map_styles.LIGHT,
        tooltip={'html': "{text}"},
    )
    st.pydeck_chart(r)


@st.cache_data
def load_temperature_geodataframe(mes: str) -> gpd.GeoDataFrame:
    gdf_temp = gpd.read_file(f'app/data/temperature_polygons_{mes}.json')
    gdf_temp = gdf_temp.set_crs('epsg:4326')
    gdf_temp['text'] = gdf_temp.temp.apply(lambda x: f"<b>Temperature :</b> {x: .1f} ºC")
    return gdf_temp
    

@st.cache_data
def load_chlorophyll_geodataframe(mes: str) -> gpd.GeoDataFrame:
    gdf_temp = gpd.read_file(f'app/data/chlorophyll_polygons_{mes}.json')
    gdf_temp = gdf_temp.set_crs('epsg:4326')
    gdf_temp['text'] = gdf_temp.chlor_raw.apply(lambda x: f"<b>Chlorophyll:</b> {x} mg/m³")
    return gdf_temp


@st.cache_data
def load_ruta() -> gpd.GeoDataFrame:
    ruta = (gpd.read_file('app/data/Track_OOO.gpx', layer='tracks')).explode().reset_index(drop=True)
    ruta = ruta.set_crs('epsg:4326')
    ruta['text'] = ruta.name.apply(lambda x: f'ruta {x}')
    return ruta


@st.cache_data    
def define_viewstate() -> pdk.ViewState: 
    return pdk.ViewState(
        **{"latitude": -39.96, "longitude": -73.7, "zoom": 10, "maxZoom": 13, "minZoom": 7, "pitch": 20, "bearing": 0}
    )


@st.cache_data
def request_gbif_api(taxonkey: int = 2440735):
    # Define GBIF API endpoint URL
    api_url = 'https://api.gbif.org/v1/occurrence/search'

    # Define query parameters
    params = {
        'taxonKey': int(taxonkey),   # GBIF taxon key for whales
        'country': 'CL',    # ISO-3166 alpha-2 country code for Chile
        'limit': 300,       # Maximum number of occurrences to retrieve per request
        'offset': 0         # Offset for pagination (starts at 0)
    }

    # Initialize empty list for occurrences
    occurrences = []
    # Loop over all pages of occurrences
    while True:
        # Send GET request to API endpoint with query parameters
        response = requests.get(api_url, params=params)
        results = response.json()['results']
        occurrences.extend(results)

        if len(results) < params['limit']:
            break

        # Increment offset for next page of occurrences
        params['offset'] += params['limit']


    df = pd.DataFrame({'gbifID': [k['gbifID'] for k in occurrences], 
                    'species': [k['species'] if 'species' in k.keys() else np.nan for k in occurrences],
                    'decimalLatitude': [k['decimalLatitude'] if 'decimalLatitude' in k.keys() else np.nan for k in occurrences],
                    'decimalLongitude': [k['decimalLongitude'] if 'decimalLongitude' in k.keys() else np.nan for k in occurrences],
                    'eventDate': [k['eventDate'] if 'eventDate' in k.keys() else np.nan for k in occurrences]})

    df = df.dropna()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['decimalLongitude'], df['decimalLatitude']), crs='EPSG:4326')
    gdf['text'] = gdf.apply(lambda x: x.species + '<br>' + x.eventDate[:10], axis=1)
    return gdf




main()

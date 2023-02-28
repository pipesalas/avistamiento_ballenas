import pandas as pd
import numpy as np
import geopandas as gpd
from config import DATES
from shapely.geometry import MultiPoint
import streamlit as st
import pydeck as pdk
import os


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

    st.title('Visualizador de las condiciones marítimas :ocean:')

    st.caption('''Para visualizar las condiciones marinas usamos la integración de ee BLABLA y agrupamos por cuadrados, sacamos la MEDIANA del mes, blabla''' )

    
    diccionario_meses = {'septiembre 2022': '2022-09-01_to_2022-10-01',
                        'octubre 2022': '2022-10-01_to_2022-11-01',
                        'noviembre 2022': '2022-11-01_to_2022-12-01',
                        'diciembre 2022': '2022-12-01_to_2023-01-01',
                        'enero 2023': '2023-01-01_to_2023-02-01',
                        'febrero 2023': '2023-02-01_to_2023-03-01',}

    col1, _ = st.columns(2)
    mes_seleccionado = col1.selectbox('Seleccionamos un mes', diccionario_meses.keys())

    # cargamos datos
    gdf_temp = load_temperature_geodataframe(diccionario_meses.get(mes_seleccionado))
    gdf_chlor = load_chlorophyll_geodataframe(diccionario_meses.get(mes_seleccionado))
    
    
    st.header('Clorofila en la superficie del oceano 🍀')
    plot_chlorophyll(gdf_chlor)
    st.caption('''Se sabe que las ballenas se alimentan de fitoplancton, que son plantas microscópicas que viven en el océano. 
    Estas plantas dependen de la luz solar y los nutrientes para crecer y, como resultado, su abundancia a menudo está relacionada con la concentración de clorofila en el agua. 
    La clorofila es un pigmento verde que ayuda a estas plantas a convertir la luz solar en energía a través de la fotosíntesis. 
    Por lo tanto, las altas concentraciones de clorofila en el agua suelen ser un indicador de la gran abundancia de fitoplancton, que a su vez puede atraer a las ballenas a la zona. 
    En otras palabras, las áreas con altas concentraciones de clorofila pueden ser buenos lugares de alimentación para las ballenas, y monitorear los niveles de clorofila puede ayudarnos
    a comprender mejor el comportamiento y la distribución de las ballenas.''')
    
    st.header('Temperatura en la superficie del oceano 🌡')
    plot_temperature(gdf_temp)
    st.caption('''La relación entre la presencia de fitoplancton y la temperatura del mar es compleja y, a menudo, depende de una variedad de factores. 
    Generalmente, el fitoplancton prospera en aguas más cálidas, pero hay muchos otros factores que entran en juego, como la disponibilidad de nutrientes, los niveles de luz y el movimiento del agua. 
    La temperatura ciertamente puede desempeñar un papel en el crecimiento y la distribución del fitoplancton, pero no es el único factor determinante.
    Del mismo modo, la relación entre la temperatura y la presencia de ballenas también es compleja. Algunas especies de ballenas prefieren aguas más frías, mientras que otras prefieren aguas más cálidas. 
    Además, el comportamiento de las ballenas puede verse influenciado por factores como la disponibilidad de presas, los patrones de apareamiento y las rutas de migración. 
    Si bien la temperatura ciertamente puede ser un factor para determinar el comportamiento de las ballenas, es solo uno de los muchos factores que deben tenerse en cuenta.''')


    st.markdown('---')
    st.markdown('---')

    tab_clor, tab_temp = st.tabs(['Clorofila', 'Temperatura'])

    with tab_clor:
        st.header('Clorofila en la superficie del oceano 🍀')
        plot_chlorophyll(gdf_chlor)
        st.caption('''Se sabe que las ballenas se alimentan de fitoplancton, que son plantas microscópicas que viven en el océano. 
        Estas plantas dependen de la luz solar y los nutrientes para crecer y, como resultado, su abundancia a menudo está relacionada con la concentración de clorofila en el agua. 
        La clorofila es un pigmento verde que ayuda a estas plantas a convertir la luz solar en energía a través de la fotosíntesis. 
        Por lo tanto, las altas concentraciones de clorofila en el agua suelen ser un indicador de la gran abundancia de fitoplancton, que a su vez puede atraer a las ballenas a la zona. 
        En otras palabras, las áreas con altas concentraciones de clorofila pueden ser buenos lugares de alimentación para las ballenas, y monitorear los niveles de clorofila puede ayudarnos
        a comprender mejor el comportamiento y la distribución de las ballenas.''')
        
    with tab_temp:
        st.header('Temperatura en la superficie del oceano 🌡')
        plot_temperature(gdf_temp)
        st.caption('''La relación entre la presencia de fitoplancton y la temperatura del mar es compleja y, a menudo, depende de una variedad de factores. 
        Generalmente, el fitoplancton prospera en aguas más cálidas, pero hay muchos otros factores que entran en juego, como la disponibilidad de nutrientes, los niveles de luz y el movimiento del agua. 
        La temperatura ciertamente puede desempeñar un papel en el crecimiento y la distribución del fitoplancton, pero no es el único factor determinante.
        Del mismo modo, la relación entre la temperatura y la presencia de ballenas también es compleja. Algunas especies de ballenas prefieren aguas más frías, mientras que otras prefieren aguas más cálidas. 
        Además, el comportamiento de las ballenas puede verse influenciado por factores como la disponibilidad de presas, los patrones de apareamiento y las rutas de migración. 
        Si bien la temperatura ciertamente puede ser un factor para determinar el comportamiento de las ballenas, es solo uno de los muchos factores que deben tenerse en cuenta.''')


    

@st.cache_data
def load_temperature_geodataframe(mes: str) -> gpd.GeoDataFrame:
    gdf_temp = gpd.read_file(f'app/data/temperature_polygons_{mes}.json')
    return gdf_temp
    

@st.cache_data
def load_chlorophyll_geodataframe(mes: str) -> gpd.GeoDataFrame:
    gdf_temp = gpd.read_file(f'app/data/chlorophyll_polygons_{mes}.json')
    return gdf_temp



@st.cache_data
def load_ruta() -> gpd.GeoDataFrame:
    ruta = (gpd.read_file('app/data/Track_OOO.gpx', layer='tracks')).explode().reset_index(drop=True)
    return ruta
        



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

    ruta = load_ruta()


    view_state = pdk.ViewState(
        **{"latitude": -39.96, "longitude": -73.7, "zoom": 10, "maxZoom": 16, "pitch": 20, "bearing": 0}
    )

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

    tooltip = {"html": "<b>Chlorophyll:</b> {chlor_raw} mg/m³"}

    layer = pdk.Layer(
                    'PathLayer',
                    data=ruta,
                    get_path='geometry.coordinates',
                    get_color=[255, 255, 0],
                    width_scale=20,
                    width_min_pixels=2,
                    get_width='width',
                    pickable=False
                    )

    r = pdk.Deck(
        [polygon_layer, layer],
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

    ruta = load_ruta()

    view_state = pdk.ViewState(
        **{"latitude": -39.96, "longitude": -73.7, "zoom": 10, "maxZoom": 16, "pitch": 20, "bearing": 0}
    )

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

    tooltip = {"html": "<b>Temperature :</b> {temp} ºC", "text": "{name}"}


    layer = pdk.Layer(
    'PathLayer',
    data=ruta,
    get_path='geometry.coordinates',
    get_color=[255, 255, 0],
    width_scale=20,
    width_min_pixels=2,
    get_width='width',
    )

    r = pdk.Deck(
        [polygon_layer, layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip=tooltip,
    )
    st.pydeck_chart(r)


main()
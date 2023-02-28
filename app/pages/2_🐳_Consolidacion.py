

import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import numpy as np
import pydeck as pdk
import plotly.express as px



def main():
    st.set_page_config(
        page_title="Vuelvealoceano",
        page_icon="whale",
        layout="wide",
    )
    # Define columns to display in table
    table_columns = ['gbifID', 'species', 'decimalLatitude', 'decimalLongitude', 'eventDate']

    # Create Streamlit app
    st.title('Whale Occurrences in Chile :whale:')
    st.write('''¡Bienvenido a nuestra aplicación Ocurrencias de Ballenas en Chile! 
    Esta aplicación le permite explorar las ocurrencias de especies de ballenas registradas en el país de Chile, utilizando datos de Global Biodiversity Information Facility (GBIF). 
    Con esta aplicación, puede ver una tabla de datos de ocurrencia para cada especie de ballena, así como un mapa de las ubicaciones donde se han observado estas especies. 
    Puede acercar y alejar el mapa y hacer clic en marcadores individuales para ver más información sobre cada ocurrencia. 
    Esta aplicación es una gran herramienta para científicos, educadores y cualquier persona interesada en aprender más sobre la distribución de las especies de ballenas en Chile.''')


    # Define list of whale species with GBIF taxon keys
    # https://www.gbif.org/species/2440483 for finding more
    whale_species = {
        'Blue whale': 2440735,
        'Fin whale': 2440718,
        'Humpback whale': 5220086,
        'Killer whale': 2440483,
        'Gray whale': 2440704
    }
    # Display table of whale occurrences

    whale_species_selection = st.selectbox('Select whale species', list(whale_species.keys()))
    

    df, gdf = request_gbif_api(whale_species.get(whale_species_selection))
    with st.expander('Datos brutos'):
        st.header('Occurrences Table')
        st.dataframe(gdf[table_columns])


    # Display map of whale occurrences
    st.header(f'Occurrences Map for {whale_species_selection} :whale2:')
    view_state = pdk.ViewState(
        latitude=-40,
        longitude=-70,
        zoom=4,
        bearing=0,
        pitch=0
    )

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=gdf,
        get_position=['decimalLongitude', 'decimalLatitude'],
        get_radius=10000,
        get_fill_color=[255, 0, 0],
        pickable=True,
        auto_highlight=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_line_color=[255, 255, 255],
        get_line_width=1,)

    tooltip = {'html': '<b>Species:</b> {species}<br><b>Event Date:</b> {eventDate}'}
    r = pdk.Deck(layers=[layer], 
                 initial_view_state=view_state, 
                 tooltip=tooltip, 
                 map_style=pdk.map_styles.LIGHT,)
    st.pydeck_chart(r)


    st.header('Avistamientos a lo largo del tiempo')
    # Create line plot of number of observations over time
    if len(gdf) > 0:
        fig = px.histogram(gdf.query('eventDate>="1990-01-01"'), x='eventDate', nbins=len(gdf), title='Number of Observations over Time')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Number of Observations')
        st.plotly_chart(fig)
    else:
        st.write('No occurrences found')


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
    return df, gdf


main()
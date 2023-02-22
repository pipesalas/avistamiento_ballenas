import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import numpy as np
import pydeck as pdk



def main():
    # Define columns to display in table
    table_columns = ['gbifID', 'species', 'decimalLatitude', 'decimalLongitude', 'eventDate']

    # Create Streamlit app
    st.title('Whale Occurrences in Chile')
    st.write('''Welcome to our Whale Occurrences in Chile app! 
    This application allows you to explore the occurrences of whale species recorded in the country of Chile, 
    using data from the Global Biodiversity Information Facility (GBIF). With this app, you can view a table 
    of occurrence data for each whale species, as well as a map of the locations where these species have 
    been observed. You can zoom in and out of the map and click on individual markers to view more information
    about each occurrence. This app is a great tool for scientists, educators, and anyone interested in 
    learning more about the distribution of whale species in Chile.''')


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
    st.header('Occurrences Map')
    view_state = pdk.ViewState(
        latitude=-30,
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
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r)



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
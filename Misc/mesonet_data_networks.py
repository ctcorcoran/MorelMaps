import os
import requests
import yaml
import time
import pandas as pd
import datetime as dt

from bs4 import BeautifulSoup
from selenium import webdriver

######################################
# MESONET CA STATIONS (w/ SOIL TEMP) #
######################################

# API KEY
API_auth = yaml.safe_load(open('../credentials.yml'))['API_auth']

# API ENDPOINT:
mesonet_meta_endpt = 'https://api.synopticdata.com/v2/stations/metadata'

# PARAMS
params = {API_auth['Mesonet']['type']:API_auth['Mesonet']['key'],
          'state':'CA', 
          'vars':'soil_temp'} 

sta_rq = requests.get(mesonet_meta_endpt,params=params)
sta_df = pd.json_normalize(sta_rq.json()['STATION'])
sta_df['MNET_ID'] = sta_df['MNET_ID'].astype(str)

# Subset for Norcal - Lat 37 = Santa Cruz
sta_df = sta_df.loc[sta_df['LATITUDE'].astype(float)>37,:]

####################################
# SYNOPTIC - MESONET NETWORK TABLE #
####################################

# Set up the Selenium WebDriver
options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=options)

# Load webpage 
mesonet_network_table_url = 'https://demos.synopticdata.com/providers/index.html'
page = driver.get(mesonet_network_table_url)

# Wait for the page to load and for JavaScript to populate the table
time.sleep(5)  

# Extract source
source = driver.page_source

# Parse HTML
soup = BeautifulSoup(source, "html.parser")

# Find table
table = soup.find('table', { 'class' : 'table-striped' , 'id':''})

# Get first table and body
thead = table.find("thead")
tbody = table.find("tbody")
    
# Extract headers
headers = [header.text for header in thead.find_all("th")]

# Extract rows
rows = []
for row in tbody.find_all("tr"):  # Skip the header row
    cells = row.find_all("td")
    rows.append([cell.text for cell in cells])

# Make Dataframe - we only really care about ID and Name columns

networks = pd.DataFrame(rows).rename(
    columns={i:headers[i] for i in range(len(headers))}
    ).drop(
        ['Record Length','Stations reporting',''],axis=1
        ).rename(columns={'ID':'MNET_ID'})
        
networks['MNET_ID'] = networks['MNET_ID'].astype(str)

################################
# MESONET NETWORKS X SOIL TEMP #
################################

sta_df = sta_df.merge(networks,on='MNET_ID')

network_counts = pd.DataFrame(sta_df['Name'].value_counts()).reset_index(drop=False)

##########
# OUTPUT #
##########

# Output network counts for nCA stations with soil temp
networks.to_csv('mesonet_network_ids.csv',index=False)

# Get date, delete old data, and  
today = dt.datetime.today().strftime('%Y-%m-%d')

prefix = 'mesonet_network_CA_soiltemp_cts_'

# Get list of old filenames in /Data/
old_filenames = [x for x in os.listdir() if x[0:len(prefix)] == prefix]

for filename in old_filenames:
    os.remove(filename)

network_counts.to_csv(prefix+today+'.csv',index=False)

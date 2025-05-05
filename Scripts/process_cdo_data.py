import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd

import datetime as dt

import time
import yaml
import utilities

# API Config
API = yaml.safe_load(open('../api_config.yml'))['API']
results_lim = 100

# Geo Config
geo_config = yaml.safe_load(open('../geo_config.yml'))

# API ENDPOINTS:
data_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
stations_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/stations'
# categories_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/datacategories'

# API DATA DATES
start = '2025-01-01'
end = dt.date.today().strftime('%Y-%m-%d')

# GEO - set buffered search extent
extent = geo_config['Extent']
buffer = 0.5 #deg
extent = {key:(val+buffer if key[4:7] == 'max' else val - buffer) for key, val in extent.items()}

# API HEADERS
headers = {API['CDO']['type']:API['CDO']['key']}

# Get Stations within Extent

params = {'extent':(str(extent['lat_min'])+','+str(extent['lon_min'])+','+str(extent['lat_max'])+','+str(extent['lon_max'])),
          'limit':results_lim,
          'datacategoryid':'TEMP',
          'start_date':start,
          'end_date':end}

req = requests.get(stations_url,
                   headers=headers,
                   params=params)
df = pd.json_normalize(req.json()['results'])

# We'll only keep stations that have data for this year

df['max_year']=[x[0] for x in df['maxdate'].str.split('-')]
df_curr = df.loc[df['max_year']==start[0:4]].reset_index(drop=True)

# Pull temp Data

params = {'datasetid':'GHCND',
          'datatypeid':'TMAX,TMIN',
          'units':'standard',
          'startdate':start,
          'limit':1000}

t_df_list = []

# Loop over stations

for sta in df_curr['id']:
    end = df_curr.loc[df_curr['id']==sta,'maxdate'].values[0]
    
    print(sta)
    print(end)
    
    # Make request and convert to dataframe
    sta_dict = {'stationid':sta,'enddate':end}
    temp_params = {**params,**sta_dict}
    temp_req = requests.get(data_url,
                            headers=headers,
                            params=temp_params)
    temp_dat = temp_req.json()
    if len(temp_dat)==0:
        print('No Data')
        continue
    temp_df = pd.json_normalize(temp_dat['results'])

    # Pivot on date to convert TMAX, TMIN to their own columns
    temp_df = temp_df.pivot(index='date',
                            columns='datatype',
                            values='value'
                            ).reset_index(
                                drop=False
                                )
    
    # Add ID back in                                                     
    temp_df['id'] = sta
    
    t_df_list.append(temp_df)

    # Sleep 1 sec to avoid overheating API
    time.sleep(1)    

# Combine
t_df = pd.concat(t_df_list).reset_index(drop=True)

# Convert/Compute Columns
t_df['date'] = pd.to_datetime(t_df['date'])     
t_df['Jday'] = t_df['date'].apply(lambda x: float(x.strftime('%j')))
t_df['Air_avg'] = (t_df['TMAX']+t_df['TMIN'])/2
t_df['Air_avg'] = t_df['Air_avg'].apply(utilities.F_to_C)

# Merge in station-specific data
t_df = t_df.merge(df_curr[['id','name','elevation','elevationUnit','longitude','latitude']],on='id')
t_df['Name_id'] = t_df['name']+' - '+t_df['id']
t_df['elevation'] *= pd.Series([1/3.28084 if x == 'FEET' else 1 for x in t_df['elevationUnit']])
t_df['elevation'] = t_df['elevation'].round(0)

# Drop and rename - prepare to save
t_df = t_df.drop(['TMAX','TMIN','elevationUnit','name','id'],axis=1).rename(
    columns={
        'date':'Date',
        'elevation':'Elev',
        'latitude':'Lat',
        'longitude':'Lon'
    })

##########
# OUTPUT #
##########

# Remove any previous versions of the dataset lying around,
# and save this version

# Get CIMIS filename prefix from config.yml
#prefix = yaml.safe_load(open('../config.yml'))['Filenames']['Mesonet']['prefix']
prefix = API['CDO']['data_filename']

# Get list of old filenames in /Data/
old_filenames = [x for x in os.listdir('../Data') if x[0:len(prefix)] == prefix]

for filename in old_filenames:
    os.remove('../Data/'+filename)

t_df.dropna().to_csv('../Data/'+prefix+'_'+dt.date.today().strftime('%Y-%m-%d')+'.csv',index=False)


#########################
# STATION LOCATION PLOTS 

fig = utilities.plot_station_locations(df_curr.rename(columns={'maxdate':'Date',
                                               'name':'Name_id',
                                               'latitude':'Lat',
                                               'longitude':'Lon'}),extent,'CDO')

fig.savefig('../Outputs/Check Plots/cdo_station_locations.png')

#################
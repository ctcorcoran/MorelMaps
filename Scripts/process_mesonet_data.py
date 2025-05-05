import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yaml
import time

import utilities

include_network_names = True

# API KEYS
API = yaml.safe_load(open('../api_config.yml'))['API']

# API ENDPOINTS:
mesonet_meta_endpt = 'https://api.synopticdata.com/v2/stations/metadata'
mesonet_timeseries_endpt = 'https://api.synopticdata.com/v2/stations/timeseries'

# API DATA DATES
start = '20250101'+'0000'
end = dt.date.today().strftime('%Y%m%d')+'0000' 

# GEO - CONVEX HULL
hull = [(40.830870, -122.787926),
        (41.436061, -120.880078),
        (38.734772, -119.602010),
        #(38.640016, -118.856148),
        (38.185355, -121.072040)]


params = {API['Mesonet']['type']:API['Mesonet']['key'],
          'state':'CA', 
          'vars':'soil_temp'} 

sta_rq = requests.get(mesonet_meta_endpt,params=params)

sta_df = pd.json_normalize(sta_rq.json()['STATION'])

sta_df['Include'] = sta_df[['LATITUDE','LONGITUDE']].astype(float).apply(lambda x: utilities.convex_hull_check(hull,(x.LATITUDE, x.LONGITUDE)), axis=1)    

# Optional: Remove MNET_ID == '66', which is CIMIS

sta_id_list = list(sta_df.loc[sta_df['Include']==True,'STID'])

print(sta_id_list)

df_list = []
all_series = []

for stid in sta_id_list:
    print(stid)
    
    params = {API['Mesonet']['type']:API['Mesonet']['key'],
              'stid':stid,
              'vars':'air_temp,soil_temp',
              'units':'temp|f',
              'start':start,
              'end':end,
              'timeformat':'%Y%m%d'}


    temp_req = requests.get(mesonet_timeseries_endpt,params=params)
    temp = temp_req.json()
    
    if len(temp['STATION'])==0:
        continue
    
    temp_df = pd.DataFrame.from_dict(temp['STATION'][0]['OBSERVATIONS'])
    
    # Append
    all_series.append(temp_df.columns)
    
    # Check if soil temp is really there
    if 'soil_temp_set_1' not in temp_df.columns:
        continue
    
    # Some stations have multiple time series for soil temp, but hard to find documentation on 
    # what makes them differ - likely depth
    temp_df = temp_df.rename({'date_time':'Date','air_temp_set_1':'air_temp','soil_temp_set_1':'soil_temp'},axis='columns')

    pivot = pd.pivot_table(data=temp_df,
                           index='Date',
                           values=['air_temp','soil_temp'],
                           aggfunc=['min','max'])
    
    pivot.columns = [x[1]+'_'+x[0] for x in pivot.columns]
    pivot = pivot.reset_index(drop=False)
    
    # Elevation unit multiplier
    if temp['STATION'][0]['UNITS']['elevation'] == 'ft':
        elev_multi = 1/3.28084
    else:
        elev_multi = 1

    pivot['Name_id'] = temp['STATION'][0]['NAME'] +'-'+ temp['STATION'][0]['STID']
    pivot['Lat'] = float(temp['STATION'][0]['LATITUDE'])
    pivot['Lon'] = float(temp['STATION'][0]['LONGITUDE'])
    if temp['STATION'][0]['ELEV_DEM'] == None:
        continue
    else:
        pivot['Elev'] = float(temp['STATION'][0]['ELEV_DEM']) * elev_multi
    pivot['Network'] = temp['STATION'][0]['MNET_ID']

    
    pivot['Air_avg'] = (pivot['air_temp_min']+pivot['air_temp_max'])/2
    pivot['Soil_avg'] = (pivot['soil_temp_min']+pivot['soil_temp_max'])/2
    
    # Convert to Celsius
    pivot['Air_avg'] = pivot['Air_avg'].apply(utilities.F_to_C)
    pivot['Soil_avg'] = pivot['Soil_avg'].apply(utilities.F_to_C)

    pivot['Date'] = pivot['Date'].apply(pd.to_datetime)
    pivot['Jday'] = pivot['Date'].apply(lambda x:int(x.strftime('%j'))) 
    
    df_list.append(pivot[['Date','Jday','Air_avg','Soil_avg','Lat','Lon','Elev','Name_id','Network']])
    
    time.sleep(1)
    
mesonet_df = pd.concat(df_list).reset_index(drop=True)

# Remove CIMIS Stations, as they are handled separately
mesonet_df = mesonet_df.loc[mesonet_df['Network']!='66',:]

# Optional: Load and add mesonet network names
if include_network_names == True:
    network_xwalk = pd.read_csv('../Misc/mesonet_network_ids.csv').rename(columns={'Name':'Network_name'})
    network_xwalk['MNET_ID'] = network_xwalk['MNET_ID'].astype(str)
    
    mesonet_df = mesonet_df.merge(network_xwalk,left_on='Network',right_on='MNET_ID').drop(['Network','MNET_ID'],axis=1)

##########
# OUTPUT #
##########

# Remove any previous versions of the dataset lying around,and save this version

# Get CIMIS filename prefix from config.yml
#prefix = yaml.safe_load(open('../config.yml'))['Filenames']['Mesonet']['prefix']
prefix = API['Mesonet']['data_filename']

# Get list of old filenames in /Data/
old_filenames = [x for x in os.listdir('../Data') if x[0:len(prefix)] == prefix]

for filename in old_filenames:
    os.remove('../Data/'+filename)

mesonet_df.dropna().to_csv('../Data/'+prefix+'_'+dt.date.today().strftime('%Y-%m-%d')+'.csv',index=False)

#########################
# STATION LOCATION PLOTS 

# Standardize station info for plotting
mesonet_df['Date'] = mesonet_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
plot_sta_df = mesonet_df.groupby(['Name_id','Lat','Lon','Network_name']).agg({'Date':'max'}).reset_index(drop=False)

# Get extent from convex hull

extent = {'lon_min':min([x[1] for x in hull]),
          'lat_min':min([x[0] for x in hull]),
          'lon_max':max([x[1] for x in hull]),
          'lat_max':max([x[0] for x in hull])}

fig = utilities.plot_station_locations(plot_sta_df,extent,'Mesonet')

fig.savefig('../Outputs/Check Plots/mesonet_station_locations.png')

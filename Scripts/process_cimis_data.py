import os
import requests
import pandas as pd
import datetime as dt
import yaml
import time

import utilities

# API KEYS
API = yaml.safe_load(open('../api_config.yml'))['API']

# API ENDPOINTS:
cimis_stations_endpt = 'http://et.water.ca.gov/api/station'
cimis_base = 'http://et.water.ca.gov/api/data' # + {endpoint}

# API DATA DATES
start = '2025-01-01'
end = dt.date.today().strftime('%Y-%m-%d') 

# API PARAMS
params = {API['CIMIS']['type']:API['CIMIS']['key'],
         'targets':'',#tgts,
         'startDate':start,
         'endDate':end,
         'dataItems':'day-air-tmp-avg,day-soil-tmp-avg'}

# GEO - CONVEX HULL
hull = [(40.830870, -122.787926),
        (41.436061, -120.880078),
        (38.734772, -119.602010),
        #(38.640016, -118.856148),
        (38.185355, -121.072040)]

### GET CIMIS STATIONS

# Get all CIMIS STATIONS
sta_rq = requests.get(cimis_stations_endpt)
sta_df = pd.json_normalize(sta_rq.json()['Stations'])
# sta_df = sta_df[sta_df['StationNbr'].isin([str(c) for c in CIMIS_foothill_stations])]

# Subset for active - string, not bool :/
sta_df = sta_df.loc[sta_df['IsActive']=='True',:]

sta_df['Lat'] = sta_df['HmsLatitude'].str.split(' / ').str[1].astype(float) #[float(x[1]) for x in sta_df['HmsLatitude'].str.split(' / ')]
sta_df['Lon'] = sta_df['HmsLongitude'].str.split(' / ').str[1].astype(float) #[float(x[1]) for x in sta_df['HmsLongitude'].str.split(' / ')]
sta_df['Name_Nbr'] = sta_df['Name'] + ' - ' + sta_df['StationNbr']

# Keep "GroundCover at some point? They're basically all grass...
sta_df = sta_df[['StationNbr','Name','Name_Nbr','Elevation','Lat','Lon']].reset_index(drop=True)

# Filter for stations in convex hull:
sta_df['Include'] = sta_df[['Lat','Lon']].apply(lambda x: utilities.convex_hull_check(hull,(x.Lat, x.Lon)), axis=1)    

### GET CIMIS DATA

# As far as I can tell, the target parameter doesn't work with a single station number
# As a work-around, I can generate lists of two (and a final list of three if need be)

sta_num_list = list(sta_df.loc[sta_df['Include']==True,'StationNbr'])

print('Stations: ',sta_num_list)

if len(sta_num_list) % 2 == 0:
    tgts = [','.join(sta_num_list[i:(i+2)]) for i in range(int(len(sta_num_list)/2))]
else:
    if len(sta_num_list) == 1:
        print('Need at least two stations')
    else:
        n_even = len(sta_num_list) - 3
        tgts = [','.join(sta_num_list[2*i:(2*(i+1))]) for i in range(int(n_even/2))]
        tgts += [','.join(sta_num_list[n_even:(len(sta_num_list)+1)])]
            

cimis_df_list = []

# loop over targets, since the CIMIS API doesn't allow a limit parameter
print('Gathering Station Data')
for tgt in tgts:
    print(tgt)
    params['targets'] = tgt
    try:
        req = requests.get(cimis_base,params=params)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        raise SystemExit(e)
    cimis_df = pd.json_normalize(req.json()['Data']['Providers'][0]['Records']).loc[:,['Date','Julian','Station','DayAirTmpAvg.Value','DaySoilTmpAvg.Value']]
    cimis_df_list.append(cimis_df)
    
    # Avoid overheating the API
    time.sleep(1)

cimis_df = pd.concat(cimis_df_list)
cimis_df['Julian'] = cimis_df['Julian'].astype(float) #pd.to_numeric(cimis_df['Julian'],downcast='float')

# Generate new station-by-station columns

lats = []
lons = []
elevs = []
name_ids = []

## This loop can be eliminated...

for sta in cimis_df['Station'].unique():
    n= len(cimis_df[cimis_df['Station']==sta])
    row = sta_df[sta_df['StationNbr']==sta]
    lats = lats + [float(row['Lat'].values[0]) for _ in range(n)]
    lons = lons + [float(row['Lon'].values[0]) for _ in range(n)]
    elevs = elevs + [float(row['Elevation'].values[0])*3.28084 for _ in range(n)] #Elevations appear to be all in ft
    name_ids = name_ids + [row['Name_Nbr'].values[0] for _ in range(n)]

cimis_df['Lat'] = lats
cimis_df['Lon'] = lons
cimis_df['Elev'] = elevs
cimis_df['Name_id'] = name_ids

cimis_df = cimis_df.rename({'DayAirTmpAvg.Value':'Air_avg','DaySoilTmpAvg.Value':'Soil_avg','Julian':'Jday'},axis='columns')

cimis_df = cimis_df.drop('Station',axis='columns')

# Convert to Celsius
cimis_df['Air_avg'] = pd.to_numeric(cimis_df['Air_avg']).apply(utilities.F_to_C)
cimis_df['Soil_avg'] = pd.to_numeric(cimis_df['Soil_avg']).apply(utilities.F_to_C)

##########
# OUTPUT #
##########

# Remove any previous versions of the dataset lying around,
# and save this version

# Get CIMIS filename prefix from config.yml
#prefix = yaml.safe_load(open('../config.yml'))['Filenames']['CIMIS']['prefix']
prefix = API['CIMIS']['data_filename']

# Get list of old filenames in /Data/
old_filenames = [x for x in os.listdir('../Data') if x[0:len(prefix)] == prefix]

for filename in old_filenames:
    os.remove('../Data/'+filename)
    
cimis_df.dropna().to_csv('../Data/'+prefix+'_'+end+'.csv',index=False)


##########################
# STATION LOCATION PLOTS 

# Standardize station info for plotting
plot_sta_df = cimis_df.groupby(['Name_id','Lat','Lon']).agg({'Date':'max'}).reset_index(drop=False)

# Get extent from convex hull

extent = {'lon_min':min([x[1] for x in hull]),
          'lat_min':min([x[0] for x in hull]),
          'lon_max':max([x[1] for x in hull]),
          'lat_max':max([x[0] for x in hull])}

fig = utilities.plot_station_locations(plot_sta_df,extent,'CIMIS')

fig.savefig('../Outputs/Check Plots/cimis_station_locations.png')

import os
import requests
import numpy as np
import pandas as pd
import datetime as dt
import yaml
import time

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
# from statsmodels.tsa.tsatools import detrend
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf, plot_accf_grid

from pygam import GAM, s, te

##########
# README #
##########

# Being able to predict soil temperature has various interesting applications,
# of particular interest is the fruiting of woodland mushroom species.
# Soil temperature can be modeled as a function of air temperature and other geospatial features.
# This file uses the California Irrigation Management Information System (CIMIS) air and soil temperature data,
# Accesed via their API (request an API here: https://et.water.ca.gov/Home/Index)

# This file will load CIMIS data, from several stations in the Northern Sierra, Foothills,
# and Modoc Plateau, and fit a GAM to the data to predict soil temperature given an imput
# air temperature time series, elevation, and latitude.

# The GAM is fit using the statsmodels package, and the partial dependence functions are
# computed using the pygam package.

# The file will also plot the data and the GAM predictions, and the partial dependence functions.

#################
# PRELIMINARIES #
#################

# API KEYS
# Store in an API config file with format:
'''
API:
  CIMIS:
    key: [your_key]
    type: appKey
'''

api_config_path = 'path/to/api_config.yml'
API = yaml.safe_load(open(api_config_path))['API']

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

# TEMPERATURE CONVERSION UTILITIES
def F_to_C(t):
    return((t-32)*(5/9))

def C_to_F(t):
    return(t*(9/5)+32)

##########################
# GET CIMIS STATION DATA #
##########################

# Get all CIMIS STATIONS
sta_rq = requests.get(cimis_stations_endpt)
sta_df = pd.json_normalize(sta_rq.json()['Stations'])
# sta_df = sta_df[sta_df['StationNbr'].isin([str(c) for c in CIMIS_foothill_stations])]

# Subset for active - string, not bool :/
sta_df = sta_df.loc[sta_df['IsActive']=='True',:]

# Parse latitude/longitude from HMS format (degrees/minutes/seconds)
sta_df['Lat'] = sta_df['HmsLatitude'].str.split(' / ').str[1].astype(float) #[float(x[1]) for x in sta_df['HmsLatitude'].str.split(' / ')]
sta_df['Lon'] = sta_df['HmsLongitude'].str.split(' / ').str[1].astype(float) #[float(x[1]) for x in sta_df['HmsLongitude'].str.split(' / ')]
sta_df['Name_Nbr'] = sta_df['Name'] + ' - ' + sta_df['StationNbr']

# Keep "GroundCover at some point? They're basically all grass...
sta_df = sta_df[['StationNbr','Name','Name_Nbr','Elevation','Lat','Lon']].reset_index(drop=True)

# Filter for stations in convex hull (requires utility functions and list hull = [(lat,lon),...]):
# sta_df['Include'] = sta_df[['Lat','Lon']].apply(lambda x: utilities.convex_hull_check(hull,(x.Lat, x.Lon)), axis=1)    

##############################
# GET CIMIS TEMPERATURE DATA #
##############################

# As far as I can tell, the target parameter doesn't work with a single station number
# As a work-around, I can generate lists of two (and a final list of three if need be)

# sta_num_list = list(sta_df.loc[sta_df['Include']==True,'StationNbr'])

# Get subset of stations if convex hull unavailable - Northern Sierra
# Selected stations span elevation gradient from foothills to high Sierra
sta_num_list = ['222','224','43','268','267','264','84','195','13','227','246','90']

print('Stations: ',sta_num_list)

# CIMIS API limitation: must request multiple stations at once
# Group stations into pairs (or triplets) for efficient API calls
if len(sta_num_list) % 2 == 0:
    tgts = [','.join(sta_num_list[2*i:(2*(i+1))]) for i in range(int(len(sta_num_list)/2))]
else:
    if len(sta_num_list) == 1:
        print('Need at least two stations')
    else:
        n_even = len(sta_num_list) - 3
        tgts = [','.join(sta_num_list[2*i:(2*(i+1))]) for i in range(int(n_even/2))]
        tgts += [','.join(sta_num_list[n_even:(len(sta_num_list)+1)])]
            
df_list = []

# loop over targets, since the CIMIS API doesn't allow a limit parameter
print('Gathering Station Data')
for tgt in tgts:
    print(tgt)
    params['targets'] = tgt
    try:
        req = requests.get(cimis_base,params=params)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        raise SystemExit(e)
    df = pd.json_normalize(req.json()['Data']['Providers'][0]['Records']).loc[:,['Date','Julian','Station','DayAirTmpAvg.Value','DaySoilTmpAvg.Value']]
    df_list.append(df)
    
    # Avoid overheating the API
    time.sleep(1)

df = pd.concat(df_list)
df['Julian'] = df['Julian'].astype(float) #pd.to_numeric(df['Julian'],downcast='float')

# Generate new station-by-station columns
# This loop can be eliminated with a more efficient merge, but works for demo

lats = []
lons = []
elevs = []
name_ids = []

## This loop can be eliminated...

for sta in df['Station'].unique():
    n= len(df[df['Station']==sta])
    row = sta_df[sta_df['StationNbr']==sta]
    lats = lats + [float(row['Lat'].values[0]) for _ in range(n)]
    lons = lons + [float(row['Lon'].values[0]) for _ in range(n)]
    elevs = elevs + [float(row['Elevation'].values[0])*3.28084 for _ in range(n)] #Elevations appear to be all in ft
    name_ids = name_ids + [row['Name_Nbr'].values[0] for _ in range(n)]

df['Lat'] = lats
df['Lon'] = lons
df['Elev'] = elevs
df['Name_id'] = name_ids

df = df.rename({'DayAirTmpAvg.Value':'Air_avg','DaySoilTmpAvg.Value':'Soil_avg','Julian':'Jday'},axis='columns')

df = df.drop('Station',axis='columns')

# Convert to Celsius - CIMIS provides data in Fahrenheit
df['Air_avg'] = pd.to_numeric(df['Air_avg']).apply(F_to_C)
df['Soil_avg'] = pd.to_numeric(df['Soil_avg']).apply(F_to_C)

#######################
# PREP DATA FOR MODEL #
#######################

# Zheng and Hunt (1993) selected an 11-day average as the best -
# Moving averages smooth daily variability and capture seasonal trends
air_window = 11

# Generate moving average for air temp, and lagged 1 day, and 
# Process each station separately to handle time series properly
temp_df_list = []

for i in range(len(df['Name_id'].unique())):
    # Subset for station
    sta = list(df['Name_id'].unique())[i]
    temp = df.loc[df['Name_id']==sta,:].reset_index(drop=True)
    
    # Add lagged air temp - captures thermal inertia of soil
    temp['Air_avg_lag1'] = temp['Air_avg'].shift(1)
    
    # Compute Moving Averages - smooth daily fluctuations
    temp['Air_mov'] = temp['Air_avg'].rolling(window=air_window,min_periods=1).mean()
    temp['Air_mov_lag1'] = temp['Air_avg'].rolling(window=air_window,min_periods=1).mean().shift(1)
    
    # Get the last time the soil was frozen-ish
    # Start analysis after soil thaws to avoid frozen soil complications
    ind = temp.loc[temp['Soil_avg'] < 0.5,:].last_valid_index()
    if ind == None:
        ind = 0
    else:
        ind +=1
    temp = temp.iloc[ind:max(temp.index),:]
    
    # Append and move forward
    temp_df_list.append(temp)
    
df = pd.concat(temp_df_list)

# Get list of stations
station_list = df['Name_id'].unique()
    
# After examining Fig 1, remove any abberant (clearly incorrect) series
# These stations show unrealistic soil-air temperature relationships
to_remove = []

df = df.loc[~df['Name_id'].isin(to_remove),:].dropna()
station_list = df['Name_id'].unique()

#################
# MODEL FITTING #
#################
    
# The GAM we fit is   
# Soil_avg ~ s(Air Temp)+s(Air Temp (Lag 1))+s(Day)+s(Air Temp,Day)+Elev+Lat #+Network

# As of 5/2/25, the Linear Unconstrained fits best, though I prefer principled constraints
# to temper the possible effects of overfitting

GAM_input_df = df[['Name_id','Soil_avg','Jday','Air_avg','Air_avg_lag1','Air_mov_lag1','Elev','Lat']]

gam_X = GAM_input_df[['Jday','Air_avg','Air_avg_lag1','Elev','Lat']]
gam_y = GAM_input_df['Soil_avg']

# Gamma GAM with principled constraints
# gam = GAM(s(0,constraints='convex')+s(1)+s(2)+s(3,constraints='monotonic_dec')+s(4,constraints='monotonic_dec')+te(0,1),
#           link='log',
#           distribution='gamma').fit(gam_X,gam_y)

# Gamma GAM
# gam = GAM(s(0)+s(1)+s(2)+s(3)+s(4)+te(0,1),
#           link='log',
#           distribution='gamma').fit(gam_X,gam_y)

# Linear GAM with principled constraints
# Constraints: convex seasonal effect, monotonic decreasing elevation/latitude effects
# These reflect physical understanding of temperature relationships
gam = GAM(s(0,constraints='convex')+s(1)+s(2)+s(3,constraints='monotonic_dec')+s(4,constraints='monotonic_dec')+te(0,1),
          link='identity',
          distribution='normal').fit(gam_X,gam_y)

# Linear GAM
# gam = GAM(s(0)+s(1)+s(2)+s(3)+s(4)+te(0,1),
#           link='identity',
#           distribution='normal').fit(gam_X,gam_y)

print('AIC: ',gam.statistics_['AIC']) # already included in summary, but hard to see in console
print(gam.summary())    

####################
# DIAGNOSTIC PLOTS #
####################

# Big Plot Array Dimensions
# (need to solve a combinatorics problem to get it as square as possible)

n_y = 4 #int(np.floor(np.sqrt(len(station_list))))
n_x = int(np.floor(len(station_list)/n_y))

########################
# Fig 1. Plot all Series

fig_as, ax_as = plt.subplots(n_x,n_y,figsize=(5*n_x,5*n_y))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] #['salmon','dodgerblue','goldenrod','forestgreen','darkorchid','magenta']

for i in range(len(station_list)):
    sta = station_list[i]
    temp = df.loc[df['Name_id']==sta,:].dropna()
    ind_i = int(np.floor(i/(n_y)))
    ind_j = int(i % n_y)
    ax_as[ind_i,ind_j].plot(temp['Jday'],
                            temp['Air_avg'],
                            marker='.',
                            linestyle='None',
                            color=colors[i % len(colors)])
    ax_as[ind_i,ind_j].plot(temp['Jday'],
                            temp['Air_mov_lag1'],
                            color=colors[i % len(colors)])
    ax_as[ind_i,ind_j].plot(temp['Jday'],
                            temp['Soil_avg'],
                            color=colors[i % len(colors)],
                            linestyle='dashed')
    ax_as[ind_i,ind_j].plot(temp['Jday'],
                            [0 for _ in temp['Jday']],
                            color='red',
                            linestyle='dashed')
    ax_as[ind_i,ind_j].set_title(sta)

# Check residuals - standardize
gam_dr = gam.deviance_residuals(gam_X,gam_y)
gam_dr = (gam_dr - stats.describe(gam_dr).mean)/np.sqrt(stats.describe(gam_dr).variance)

#############################
# Fig 2. Residual-based Plots

fig_r, ax_r = plt.subplots(2,2,figsize=(12,12))

# Residual Histogram
ax_r[0,0].hist(gam_dr)
ax_r[0,0].set_title('Histogram of Residuals')

# Q-Q Residual Plot
# **If aberrant stations are not removed, left tail will be heavy and show up here
sm.qqplot(gam_dr,dist=stats.norm,ax = ax_r[0,1],line='45')
ax_r[0,1].set_title('Q-Q Plot - Residuals')

# Residuals vs Predictors
ax_r[1,0].plot(df['Jday'],gam_dr,marker='.',linestyle='None')
ax_r[1,0].set_title('Residuals vs. Jday')

# Response 
ax_r[1,1].plot(gam.predict(gam_X),gam_y,marker='.',linestyle='None')
ax_r[1,1].set_title('GAM Prediction vs. GAM Observation')

#############################################
# Fig 3. Partial dependence functions for GAM

n_terms = len(gam.terms)-1
fig_p = plt.figure(figsize=(24,6))

var_dict = {0:'Jday',1:'Air_avg',2:'Air_avg_lag1',3:'Elev',4:'Lat'}

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    if repr(term)[0:2] == 'te':
        # Tensor product term (interaction) - plot as 3D surface
        XX = gam.generate_X_grid(term=i,meshgrid=True)
        pdep = gam.partial_dependence(term=i, X=XX, meshgrid=True)
        ax_p = fig_p.add_subplot(1,n_terms,i+1,projection='3d')
        ax_p.plot_surface(XX[0], XX[1], pdep, cmap='viridis')
    else:
        # Univariate smooth term - plot with confidence intervals
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax_p = fig_p.add_subplot(1,n_terms,i+1)
        ax_p.plot(XX[:, term.feature], pdep)
        ax_p.plot(XX[:, term.feature], confi, c='r', ls='--')
    term_split = repr(term).split('(')
    ax_p.set_title(var_dict[int(term_split[1][0])])

#############################
# Fig 4. Plot all GAM results

def prediction_intervals(gam_X,gam_y,gam,n_boot=1,window=10):
    # Sample 1000 new predictions by fitting 1 model on bootstraps of the data
    # This provides uncertainty quantification for the GAM predictions
    samples = gam.sample(gam_X, 
                         gam_y, 
                         quantity='y', 
                         n_draws=1000, 
                         sample_at_X=gam_X, 
                         n_bootstraps=n_boot)
    
    # Compute cumulative sum and moving average of each sample
    # These are key metrics for ecological applications (degree days, etc.)
    cum_samples = np.array([list(pd.Series(x).cumsum()) for x in samples])
    mov_samples = np.array([list(pd.Series(x).rolling(window=window,min_periods=1).mean()) for x in samples])

    # Compute percentiles of the sampled data and cumulative sampled data
    q = [2.5, 97.5]
    percentiles = np.percentile(samples, q=q, axis=0).T
    percentiles =  [[x[0] for x in percentiles],[x[1] for x in percentiles]]    
    
    cum_percentiles = np.percentile(cum_samples, q=q, axis=0).T
    cum_percentiles =  [[x[0] for x in cum_percentiles],[x[1] for x in cum_percentiles]]
    
    # Generate median and bounds 
    #qq = [2.5,50.0,97.5]
    mov_percentiles = np.percentile(mov_samples, q=q, axis=0).T
    mov_percentiles =  [[x[0] for x in mov_percentiles],[x[1] for x in mov_percentiles]]
    
    return({'cumulative_samples':cum_samples,
            'mov_samples':mov_samples,
            'prediction_interval':percentiles,
            'cumulative_prediction_interval':cum_percentiles,
            'moving_prediction_interval':mov_percentiles})


# Define plotting function to handle
def plot_GAM_prediction(sta,ax,df,gam):
    # Make prediction and compute intervals
    temp = df.loc[df['Name_id']==sta,:].dropna()
    gam_in = temp[['Jday','Air_avg','Air_avg_lag1','Elev','Lat']]
    gam_out = gam.predict(gam_in)
    gam_ci = prediction_intervals(gam_in,temp['Soil_avg'],gam,n_boot=1)
    
    # Plot prediction
    ax.plot(gam_in['Jday'],gam_out,color='forestgreen',marker='.')
    ax.fill_between(gam_in['Jday'],
                    gam_ci['prediction_interval'][0],
                    gam_ci['prediction_interval'][1],
                    color='forestgreen',
                    alpha=0.5)
    
    # Moving Prediction
    gam_out_mov = pd.Series(gam_out).rolling(window=7,min_periods=1).mean()
    ax.plot(gam_in['Jday'],gam_out_mov,color='salmon')
    ax.fill_between(gam_in['Jday'],
                    gam_ci['moving_prediction_interval'][0],
                    gam_ci['moving_prediction_interval'][1],
                    color='salmon',
                    alpha=0.5)
    
    # Plot observed data
    ax.plot(gam_in['Jday'],temp['Soil_avg'],
            color='brown',
            marker='.',
            linestyle='None'
            )
    ax.plot(gam_in['Jday'],temp['Soil_avg'].rolling(window=10,min_periods=1).mean(),
            color='brown',
            linestyle='dashed'
            )
    ax.plot(gam_in['Jday'],temp['Air_avg'],
            color='dodgerblue',
            marker='.',
            linestyle='None'
            )
    ax.plot(gam_in['Jday'],
            temp['Air_mov_lag1'],
            color='dodgerblue',
            linestyle='dashed'
            )
    
    # Set title
    ax.set_title(sta+" ("+str(temp['Elev'].unique()[0])+" )")
    
    return(ax)

fig_g, ax_g = plt.subplots(n_x,n_y,figsize=(10*n_x,10*n_y))

for i in range(len(station_list)):
    #
    ind_i = int(np.floor(i/(n_y)))
    ind_j = int(i % n_y)
    #
    sta = station_list[i]

    ax_g[ind_i,ind_j] = plot_GAM_prediction(sta,ax_g[ind_i,ind_j],df,gam)

###############################################################################
# EXTRA PLOTS: CORRELATION FUNCTIONS - need to detrend the data first
# plot_pacf(df.loc[df['Name_id']==df['Name_id'].unique()[0],'Soil_avg'])

# plot_accf_grid(df.loc[df['Name_id']==df['Name_id'].unique()[0],['Air_mov_lag1','Soil_avg']].dropna())
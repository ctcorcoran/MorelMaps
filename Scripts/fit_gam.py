import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
# from statsmodels.tsa.tsatools import detrend
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf, plot_accf_grid

from pygam import GAM, s, te
import pickle

import utilities

# GAM output filename
gam_filename = '../Data/gam.pkl'

# Load data files
df_list= []
for filename in os.listdir('../Data'):
    if filename.split('_')[0] in ['cimis','mesonet']:
        print(filename)
        temp = pd.read_csv('../Data/'+filename)
        if filename.split('_')[0] == 'cimis':
            temp['Network_name'] = 'CIMIS California Irrigation Management Information System'
        df_list.append(temp)

df = pd.concat(df_list).reset_index()

# Encode network name as integer for model
df['Network_int'] = df['Network_name'].factorize()[0]

#######################
# PREP DATA FOR MODEL #
#######################

# Zheng and Hunt (1993) selected an 11-day average as the best, I use 10 as the window
# across these scripts for some reason...
air_window = 10

# Generate moving average for air temp, and lagged 1 day, and 
temp_df_list = []

for i in range(len(df['Name_id'].unique())):
    # Subset for station
    sta = list(df['Name_id'].unique())[i]
    temp = df.loc[df['Name_id']==sta,:].reset_index(drop=True)
    
    # Add lagged air temp
    temp['Air_avg_lag1'] = temp['Air_avg'].shift(1)
    
    # Compute Moving Averages
    temp['Air_mov'] = temp['Air_avg'].rolling(window=air_window,min_periods=1).mean()
    temp['Air_mov_lag1'] = temp['Air_avg'].rolling(window=air_window,min_periods=1).mean().shift(1)
    
    # Get the last time the soil was frozen-ish
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
to_remove = ['Chickering American River Reserve (UCNRS)-UCCA',
             '619Line-Sierra Valley West-LIB10',
             'Skyline Harvest-C3SKY',
             'West Incline Village-NV028',
             'BKY4201-Summit-LIB04',
             'POR31-Iron Horse-LIB09',
             'ONION CREEK NEAR SERENE LAKES 3SE-ONCC1']

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
gam = GAM(s(0,constraints='convex')+s(1)+s(2)+s(3,constraints='monotonic_dec')+s(4,constraints='monotonic_dec')+te(0,1),
          link='identity',
          distribution='normal').fit(gam_X,gam_y)

# Linear GAM
# gam = GAM(s(0)+s(1)+s(2)+s(3)+s(4)+te(0,1),
#           link='identity',
#           distribution='normal').fit(gam_X,gam_y)

print('AIC: ',gam.statistics_['AIC']) # already included in summary, but hard to see in console
print(gam.summary())    

# SAVE GAM - .PKL
 
with open(gam_filename, 'wb') as file:
    pickle.dump(gam, file)

####################
# DIAGNOSTIC PLOTS #
####################

# Big Plot Array Dimensions
# (need to solve a combinatorics problem to get it as square as possible)

n_y = 6 #int(np.floor(np.sqrt(len(station_list))))
n_x = int(np.floor(len(station_list)/n_y))+1

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
        XX = gam.generate_X_grid(term=i,meshgrid=True)
        pdep = gam.partial_dependence(term=i, X=XX, meshgrid=True)
        ax_p = fig_p.add_subplot(1,n_terms,i+1,projection='3d')
        ax_p.plot_surface(XX[0], XX[1], pdep, cmap='viridis')
    else:
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax_p = fig_p.add_subplot(1,n_terms,i+1)
        ax_p.plot(XX[:, term.feature], pdep)
        ax_p.plot(XX[:, term.feature], confi, c='r', ls='--')
    term_split = repr(term).split('(')
    ax_p.set_title(var_dict[int(term_split[1][0])])

#############################
# Fig 4. Plot all GAM results

fig_g, ax_g = plt.subplots(n_x,n_y,figsize=(10*n_x,10*n_y))

for i in range(len(station_list)):
    #
    ind_i = int(np.floor(i/(n_y)))
    ind_j = int(i % n_y)
    #
    sta = station_list[i]

    ax_g[ind_i,ind_j] = utilities.plot_GAM_prediction(sta,ax_g[ind_i,ind_j],df,gam)


##############
# SAVE PLOTS #
##############

fig_as.savefig('../Outputs/Check Plots/gam_input_all_series.png')
fig_r.savefig('../Outputs/Check Plots/gam_residual_diagnostics.png')
fig_p.savefig('../Outputs/Check Plots/gam_partial_dependence.png')
fig_g.savefig('../Outputs/Check Plots/gam_output_all_series.png')

###############################################################################
# View single station's GAM predictions
# sta_num = 40

# fig_s, ax_s = plt.subplots(2,1)

# ax_s[0] = utilities.plot_GAM_prediction(station_list[sta_num],ax_s[0],df,gam)
# ax_s[1] = utilities.plot_GAM_cumulative_prediction(station_list[sta_num],ax_s[1],df,gam)


# CORRELATION FUNCTIONS - need to detrend the data first
# plot_pacf(df.loc[df['Name_id']==df['Name_id'].unique()[0],'Soil_avg'])

# plot_accf_grid(df.loc[df['Name_id']==df['Name_id'].unique()[0],['Air_mov_lag1','Soil_avg']].dropna())
# 
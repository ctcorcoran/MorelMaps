import os
import numpy as np
import pandas as pd
import pickle
import time

import utilities

# Load GAM
gam_filepath = '../Data/gam.pkl'
with open(gam_filepath, 'rb') as file:
    gam = pickle.load(file)

# Load prediction domain points
domain_df = pd.read_csv('../Data/prediction_domain_pts.csv')

# Load local temperature data 
t_df_list= []

for filename in os.listdir('../Data'):
    if filename.split('_')[0] in ['cdo']:
        temp = pd.read_csv('../Data/'+filename)
        t_df_list.append(temp)

t_df = pd.concat(t_df_list).reset_index(drop=True)

# Set output filepath
output_filepath = '../Outputs/Results/model_results.csv'

###################################
# 1. Interpolate Air Temperatures #
###################################

# Get station spatial data
stations = t_df[['Name_id','Lat','Lon']].drop_duplicates()

# Convert temps to Kelvin, then to potential temps for interpolation
t_df['Air_avg_K'] = t_df['Air_avg'].apply(utilities.C_to_K)
t_df['Air_avg_K_pot'] = t_df.apply(lambda x: utilities.temp_act_to_pot(x['Air_avg_K'],x['Elev']),axis=1)

t_pot_df = t_df.pivot(index='Jday',columns='Name_id',values='Air_avg_K_pot').fillna(0.0)

def interpolate_temp(lon,lat,elev):
    # Get distances to stations
    dists = utilities.compute_distances(lon,lat,stations[['Name_id','Lon','Lat']].set_index('Name_id'))
    
    # First dot product gives temps by Jday 1/dist^2 weighted average, and the second
    # gives the normalizing constant by converting each nonzero value to 1 and then
    # dotting with 1/dist^2
    t_interp = t_pot_df.dot(1/dists**2)/t_pot_df.astype(bool).astype(int).dot(1/dists**2)
    
    # Convert potentials back to actuals and to Celsius 
    t_interp = t_interp.apply(lambda x: utilities.temp_pot_to_act(x,elev)).apply(utilities.K_to_C)
    
    return(t_interp)
   

################################
# 2. Predict Soil Temperatures #
################################
    
def make_gam_input(x):
    lon = x['lon']
    lat = x['lat']
    elev = x['elev']
    
    # GAM input order = 'Jday','Air_avg','Air_avg_lag1','Elev','Lat'
    temp = pd.DataFrame(interpolate_temp(lon,lat,elev)).rename(columns={0:'Air_avg'}).reset_index(drop=False)
    temp['Air_avg_lag1'] = temp['Air_avg'].shift(1)
    temp.loc[0,'Air_avg_lag1'] = temp.loc[0,'Air_avg']
    temp['Elev'] = elev
    temp['Lat'] = lat
    
    return(temp)

results_list = []
n_boot = 1
window = 10
    
tic = time.perf_counter()
pct_complete = 0.0

for i in domain_df.index:
    # Predict soil temp with GAM
    gam_in = make_gam_input(domain_df.loc[i,:])
    gam_out = [max(x,0.0) for x in gam.predict(gam_in)] #Predicted soil temp < 0 is frozen
    
    # print(i,max(gam_in['Air_avg']))
    
    # Combine inputs and output and compute moving avg and cumulative sum
    results = gam_in.join(pd.DataFrame({'Lon':domain_df.loc[i,'lon'],'Soil_avg':gam_out}))
    results['Soil_mov'] = results['Soil_avg'].rolling(window=window,min_periods=1).mean()
    results['Soil_cum'] = results['Soil_avg'].cumsum()

    # Resample predictions
    resamp = utilities.prediction_intervals(gam_in,gam_out,gam,n_boot,window)
        
    # Add to intervals to results
    results = results.join(pd.DataFrame({'Soil_avg_L':resamp['prediction_interval'][0],
                                         'Soil_avg_U':resamp['prediction_interval'][1],
                                         'Soil_mov_L':resamp['moving_prediction_interval'][0],
                                         'Soil_mov_U':resamp['moving_prediction_interval'][1],
                                         'Soil_cum_L':resamp['cumulative_prediction_interval'][0],
                                         'Soil_cum_U':resamp['cumulative_prediction_interval'][1]}))
    
    # 10-day Soil Moving Average > 10.0C
    results['Soil_mov_thresh_prob'] = np.apply_along_axis(lambda x: len(x[x>10.0])/len(x),1,resamp['mov_samples'].T)
    
    # Cumulative > Thresh 
    results['Soil_cum_thresh_prob'] = np.apply_along_axis(utilities.prob_convolution_score,1,resamp['cumulative_samples'].T)
    
    # Add index as unique estimation point id
    results['Point_id'] = i
    
    # Append Results:
    results_list.append(results)
    
    # Report some loop timing estimates:
    time_avg_period = 10
    if i == time_avg_period-1:
        toc = time.perf_counter()
        t_iter = (toc-tic)/time_avg_period
        print('Time per iteration: ',t_iter)
        print('Estimated time to complete: ',np.round(np.floor(t_iter*len(domain_df)/60),2), ' minutes')

    # Report progress
    if np.floor(10*(i+1)/len(domain_df))*10-pct_complete > 0.0:
        toc = time.perf_counter()
        pct_complete = np.floor(10*(i+1)/len(domain_df))*10
        print(pct_complete,' % Complete - ',np.floor((toc-tic)/60), ' minutes ',np.round((toc-tic) %60,2), ' seconds elapsed')

toc = time.perf_counter()

print('Elapsed Time: ',np.floor((toc-tic)/60), ' minutes ',np.round((toc-tic) %60,2), ' seconds')

results_df = pd.concat(results_list)  

# Add date column back in
results_df = results_df.merge(t_df[['Date','Jday']].drop_duplicates(),on='Jday')

# Write results to file
results_df.to_csv(output_filepath,index=False)

###################
# DIAGNOSTIC PLOTS

# fig, ax = plt.subplots(1,1)

# sns.scatterplot(data=results_df,x='Jday',y='Air_avg',color='gray',alpha=0.5)
# sns.scatterplot(data=t_df,x='Jday',y='Air_avg',hue='Name_id')

# fig2, ax2 = plt.subplots(1,1)

# results_df['Air_avg_K_pot'] = results_df.apply(lambda x: utilities.temp_act_to_pot(utilities.C_to_K(x['Air_avg']),x['Elev']),axis=1)

# #sns.scatterplot(data=t_df,x='Jday',y='Air_avg_K_pot',color='gray',alpha=0.5)
# ax2.scatter(results_df['Jday'],results_df['Air_avg_K_pot'],color='black')

# sns.scatterplot(data=t_df,x='Jday',y='Air_avg_K_pot',hue='Name_id')

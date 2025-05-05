import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd

#####################
# UTILITY FUNCTIONS #
#####################

#############
# TEMPERATURE

# Conversions

def F_to_C(t):
    return((t-32)*(5/9))

def C_to_F(t):
    return(t*(9/5)+32)

#

def F_to_K(t):
    return((t-32)*(5/9)+273.15)

def K_to_F(t):
    return((t-273.15)*(9/5)+32)

#

def C_to_K(t):
    return(t+273.15)

def K_to_C(t):
    return(t-273.15)

# Actual to Potential

T_b = 300 #K
lamb = -0.0065 #K m^{-1}
C_p = 1005 #J kg^{-1} K^{-1}
g = 9.80616 #m s^{-2}

def temp_act_to_pot(T_a,z):
    return(T_a*(T_b/(T_b+lamb*z))**(-g/(lamb*C_p)))

def temp_pot_to_act(Theta_a,z):
    return(Theta_a*(T_b/(T_b+lamb*z))**(g/(lamb*C_p)))

##########
# DISTANCE

bar_R = 6371.009 #in kilometers
const = np.pi/180

def great_circle_dist(lon1,lat1,lon2,lat2):
    delta_lambda = np.abs(lon2-lon1)
    delta_phi = np.abs(lat2-lat1)
    root = np.sqrt(np.sin(const*delta_phi/2)**2+(1-np.sin(const*delta_phi/2)**2-np.sin(const*(lat1+lat2)/2)**2)*(np.sin(const*delta_lambda/2)**2))
    return(max(2*bar_R*np.arcsin(root),1e-6)) #Removes issues of dividing by zero

# Compute Distances (km) between point (lon,lat) and a dataframe of other points
def compute_distances(lon,lat,lon_lat_df):
    dists = lon_lat_df.apply(lambda x: great_circle_dist(x['Lon'], x['Lat'],lon,lat), axis=1)
    return(dists)

##########################################
# GAM RESAMPLING, UNCERTAINTY, AND SCORING

def prediction_intervals(gam_X,gam_y,gam,n_boot=1,window=10):
    # Sample 1000 new predictions by fitting 1 model on bootstraps of the data
    samples = gam.sample(gam_X, 
                         gam_y, 
                         quantity='y', 
                         n_draws=1000, 
                         sample_at_X=gam_X, 
                         n_bootstraps=n_boot)
    
    # Compute cumulative sum and moving average of each sample
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

# Degree Day Threshold
X = [365,435,441,477,583] #Mihail 2007 obs. deg. days

# MLEs of Inverse Gaussian parameters
mu_hat = np.mean(X)
lambda_hat = (sum([1/x for x in X])/len(X)-1/mu_hat)**(-1)

# Generate samples and distr for IG threshold 
thresh_samp = stats.invgauss.rvs(mu = mu_hat/lambda_hat, loc = 0, scale = lambda_hat, size=10000)
thresh_dist = np.histogram(thresh_samp,
                       bins=100,
                       density=True)

# Probability that cumulative soil temp > threshold, when threshold is unknown,
# is a convolution - we'll approximate that here with a discrete 

def prob_convolution_score(x):
    const_thresh = sum(thresh_dist[0])
    const_x = len(x)
    # int_0^infty P(cum_soil_temp > a)*p(a) da
    return(sum([(len(x[x>thresh_dist[1][i]])/const_x)*(thresh_dist[0][i]/const_thresh) for i in range(len(thresh_dist[0]))]))

###########
# PLOTTING

# STATION LOCATIONS

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_station_locations(sta_df,extent,dataset_title):
    # Check for Network_name column (Mesonet only)
    if 'Network_name' not in sta_df.columns:
        sta_df['Network_name'] = dataset_title
    
    # Get fire perimeter
    perim_filepath = '../Data/GIS/Park Fire Perimeter 2024/Park_Fire_Perimeter_2024.shp'
    perim = gpd.read_file(perim_filepath).to_crs('EPSG:4326')
    perim_poly = max(perim.geometry[0].geoms, key = lambda x: x.area)

    # Background Plot
    rq = cimgt.OSM()

    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection=rq.crs)

    ax.set_extent([extent[x] for x in ['lon_min','lon_max','lat_min','lat_max']])
    ax.add_image(rq, 10)

    # Plot Fire Perimeter
    plt.plot(*perim_poly.exterior.xy,
             transform=ccrs.PlateCarree(),
             color='black')

    # Add Stations
    ax.scatter(sta_df['Lon'],
               sta_df['Lat'],
               transform=ccrs.PlateCarree(),
               color=sta_df['Network_name'].replace(sta_df['Network_name'].unique(),
                                                  list(range(len(sta_df['Network_name'].unique())))
                                                  ).apply(lambda x: colors[x]),
               marker='o')
    
    # Title
    ax.set_title('Station Locations (and Last Update) - '+dataset_title)

    for i in range(len(sta_df)):
        ax.annotate(sta_df.loc[i,'Name_id']+' - ' +sta_df.loc[i,'Date'],
                    xy=(sta_df.loc[i,'Lon'],sta_df.loc[i,'Lat']),
                    transform=ccrs.PlateCarree())
        
    return(fig)

# GAM PLOTTING

def plot_GAM_cumulative_prediction(sta,ax,df,gam):
    # Make prediction and compute intervals
    temp = df.loc[df['Name_id']==sta,:].dropna()
    gam_in = temp[['Jday','Air_avg','Air_avg_lag1','Elev','Lat']]
    gam_out = gam.predict(gam_in)
    gam_ci = prediction_intervals(gam_in,temp['Soil_avg'],gam,n_boot=1)
    
    # Plot cumulative prediction
    ax.plot(gam_in['Jday'],temp['Soil_avg'].cumsum(),
            color='brown',
            marker='.',
            linestyle='None')
    ax.plot(gam_in['Jday'],
            gam_out.cumsum(),
            color='brown',
            linestyle='dashed')
    ax.fill_between(gam_in['Jday'],
                    gam_ci['cumulative_prediction_interval'][0],
                    gam_ci['cumulative_prediction_interval'][1],
                    color='brown',
                    alpha=0.5)
    return(ax)

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


# RESULTS PLOTTING

plot_var_dict = {'Soil_avg': 'Soil Temp (C)',
                 'Soil_mov': 'Soil Temp (C) - 10 day moving average', 
                 'Soil_cum': 'Soil Temp (C) - Cumulative', 
                 'Soil_mov_thresh_prob':'Prob(Soil Temp 10 day moving average > 10.0 C)', 
                 'Soil_cum_thresh_prob':'Prob(Cumulative Soil Temp > Threshold)'}


def plot_results(plot_date,plot_var,results,geoms):
    # Compute X=lon, Y=lat, Z to plot (lon,lat,Z)
    X = results.loc[results['Date']==plot_date,'Lon']
    Y = results.loc[results['Date']==plot_date,'Lat']
    Z = results.loc[results['Date']==plot_date,plot_var]
    
    # Set Background Map
    rq = cimgt.OSM() #rq = cimgt.Stamen('terrain-background')
    
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection=rq.crs)
    
    ax.set_extent(geoms['extent'])
    ax.add_image(rq, 10)
    
    # Add contour
    # X_, Y_ = np.meshgrid(X,Y)
    
    # Even though we have exact values at (x,y), we use an interpolator
    # over a triangulation to easily generate the Z array: 
    # interpolator = tri.LinearTriInterpolator(tri.Triangulation(X, Y),Z)
    # Z_ = interpolator(X_,Y_)
    
    # cont = ax.contourf(X_,Y_,Z_,
    #                       transform=ccrs.PlateCarree(),
    #                       cmap='plasma',
    #                       alpha=0.5)
    
    levels = np.linspace(Z.min(),Z.max(),10)
    
    # Some issues with the contour when Z is at max or min, so here is my workaround
    Z[Z==Z.min()] = Z[Z==Z.min()]+1e-8
    Z[Z==Z.max()] = Z[Z==Z.max()]-1e-8
    
    cont = ax.tricontourf(X,Y,Z,
                       transform=ccrs.PlateCarree(),
                       cmap='plasma',
                       levels=levels,
                       alpha=0.5)
    
    # CLIP - 
    # Convert polygon to pacth
    # perim_path = mpath.Path(list(perim_poly.exterior.coords))
    # perim_patch = patches.PathPatch(perim_path,
    #                                 transform=ccrs.PlateCarree(),#ax.transData, 
    #                                 facecolor="none", 
    #                                 edgecolor="none")
    
    # cont.set_clip_path(perim_patch)
    
    # Add Land
    geoms['fs_land'].loc[1:2].boundary.plot(ax=ax,
                 transform=ccrs.PlateCarree(),
                 color='black',
                 #linestyle='dotted',
                 linewidth=2
                 )
    
    # Add Roads
    geoms['fs_roads'].plot(ax=ax,
                  transform=ccrs.PlateCarree(),
                  color='white',
                  aspect=1,
                  #alpha=0.5
                  )
    
    # Add Fire Perimeter
    ax.plot(*geoms['perim_poly'].exterior.xy,
             transform=ccrs.PlateCarree(),
             color='black',
             linewidth=5)
    
    
    plt.colorbar(cont,ax=ax)

    ax.set_title(plot_var_dict[plot_var]+' ('+plot_date.strftime('%Y-%m-%d')+')',fontsize=16)

    return(fig)

#############
# CONVEX HULL

def cross_product(o, a, b):
    # Compute the cross product of vector OA and OB.

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def graham_scan(points):
    # Compute the convex hull of a set of 2D points using the Graham Scan algorithm.
    # Sort the points lexicographically (by x, then by y)
    points = sorted(points)

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it is repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]

def convex_hull_check(points, test_point):
    # Check whether a point is inside the convex hull of a set of points.

    convex_hull = graham_scan(points)

    # Test if the point is inside the convex hull
    for i in range(len(convex_hull)):
        p1 = convex_hull[i]
        p2 = convex_hull[(i + 1) % len(convex_hull)]
        if cross_product(p1, p2, test_point) < 0:
            return False  # Outside the convex hull
    return True

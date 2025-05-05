import yaml
import pandas as pd
import geopandas as gpd

import utilities

# Get Extent from geo_config
geo_config = yaml.safe_load(open('../geo_config.yml'))
extent_dict = geo_config['Extent']
buffer = 0.1 # in deg
extent = [extent_dict[key] + buffer if key[4:7] == 'max' else extent_dict[key] - buffer for key in ['lon_min','lon_max','lat_min','lat_max']]

# Shapefile paths
perim_filepath = '../Data/GIS/Park Fire Perimeter 2024/Park_Fire_Perimeter_2024.shp'
roads_filepath = '../Data/GIS/FS Roads/fs_roads.shp'
land_filepath = '../Data/GIS/FS Land/fs_land.shp'

# results path
results_filepath = '../Data/model_results.csv'

# Read in shape files
perim = gpd.read_file(perim_filepath).to_crs('EPSG:4326')
fs_roads = gpd.read_file(roads_filepath)
fs_land = gpd.read_file(land_filepath)

# Extract just the largest fire perimeter polygon (I should have done this earlier...)
perim_poly = max(perim.geometry[0].geoms, key = lambda x: x.area)

# Read in results
results = pd.read_csv(results_filepath).drop_duplicates() #Drop dupes just in case
results['Date'] = pd.to_datetime(results['Date'])

# Collect geometries to pass to plot_results()
geoms = {'extent':extent,
         'fs_roads':fs_roads,
         'fs_land':fs_land,
         'perim_poly':perim_poly}

##############
# MAKE PLOTS #
##############

# Plot settings
plot_date = max(results['Date'])  #dt.datetime.strptime('2025-03-01','%Y-%m-%d')
#plot_var = #'Soil_avg','Soil_mov', 'Soil_cum', 'Soil_mov_thresh_prob', 'Soil_cum_thresh_prob'

for plot_var in ['Soil_avg','Soil_mov', 'Soil_cum', 'Soil_mov_thresh_prob', 'Soil_cum_thresh_prob']:
    fig = utilities.plot_results(plot_date,plot_var,results,geoms)
    fig.savefig('../Outputs/Results/'+plot_var+'_'+plot_date.strftime('%Y-%m-%d'))
    
    fig.show()


import yaml
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import rasterio as rio
from rasterio.plot import show

# Set output filepath
output_filepath = '../Data/prediction_domain_pts.csv'

# Get Extent from geo_config
geo_config = yaml.safe_load(open('../geo_config.yml'))
extent_dict = geo_config['Extent']
buffer = 0.1 # in deg
extent = [extent_dict[key] + buffer if key[4:7] == 'max' else extent_dict[key] - buffer for key in ['lon_min','lon_max','lat_min','lat_max']]

# Shapefile paths
perim_filepath = '../Data/GIS/Park Fire Perimeter 2024/Park_Fire_Perimeter_2024.shp'
roads_filepath = '../Data/GIS/FS Roads/fs_roads.shp'
land_filepath = '../Data/GIS/FS Land/fs_land.shp'

# DEM path
dem_filepath = '../Data/GIS/dem.tif'

# Read in shape files
perim = gpd.read_file(perim_filepath).to_crs('EPSG:4326')
fs_roads = gpd.read_file(roads_filepath)
fs_land = gpd.read_file(land_filepath)

# Extract just the largest fire perimeter polygon (I should have done this earlier...)
perim_poly = max(perim.geometry[0].geoms, key = lambda x: x.area)

# Generate Mesh (pts_df) for extent
n_lat = 50
n_lon = 50

lats = [extent_dict['lat_min']+(i/n_lat)*(extent_dict['lat_max']-extent_dict['lat_min']) for i in range(n_lat+1)]
lons = [extent_dict['lon_min']+(i/n_lat)*(extent_dict['lon_max']-extent_dict['lon_min']) for i in range(n_lon+1)]
pts_df = pd.DataFrame([(lon,lat) for lon in lons for lat in lats]).rename(columns={0:'lon',1:'lat'})

# Convert pts mesh dataframe to  geodataframe
pts_gdf = gpd.GeoDataFrame(pts_df, geometry=gpd.points_from_xy(pts_df.lon, pts_df.lat))

# Get indices of pts within the perimeter
pts_mask = pts_gdf.within(perim_poly) 

# Subset the mesh for interior points
pts_interior = pts_df.loc[pts_mask].reset_index(drop=True)
pts_interior_gdf = pts_gdf.loc[pts_mask].reset_index(drop=True)

# Open DEM temporarily to extract elevations for interior points
with rio.open(dem_filepath,'r') as dem:
    dem_array = dem.read(1).astype('float64')
    dem_trans = dem.transform

# Get DEM array positions for interior points
rows, cols = rio.transform.rowcol(dem_trans, pts_interior['lon'], pts_interior['lat'])

# Add elevation column (m)
pts_interior.loc[:,'elev'] = [dem_array[rows[i],cols[i]] for i in range(len(pts_interior['lat']))] #Meters to Feet
pts_interior_gdf['elev'] = pts_interior['elev']
    
pts_interior.to_csv(output_filepath)

#########
# PLOTS #
#########

# Fig. 1 - Prediction Domain Points

# Set Background Map
rq = cimgt.OSM() #rcimgt.Stamen('terrain-background')

fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection=rq.crs)

ax.set_extent(extent)
ax.add_image(rq, 10)

fs_land.loc[1:2].boundary.plot(ax=ax,
             transform=ccrs.PlateCarree(),
             color='dodgerblue',
             linewidth=1
             )

fs_roads.plot(ax=ax,
              transform=ccrs.PlateCarree(),
              color='brown',
              aspect=1,
              alpha=0.5
              )

pts_interior_gdf.plot(ax=ax,
                      transform=ccrs.PlateCarree(),
                      markersize=1)

ax.plot(*perim_poly.exterior.xy,
         transform=ccrs.PlateCarree(),
         color='black')

fig.savefig('../Outputs/Check Plots/prediction_domain.png')

# Fig 2 - Test DEM query

# fig2, ax2 = plt.subplots(1,1,figsize=(12,12))

# show(dem_array, #np.ma.masked_where(dem_out==0,dem_out),
#      transform=dem_trans,
#      ax=ax2,
#      alpha=1,
#      cmap='Greys')

# pts_interior_gdf.plot(column='elev',
#                       ax=ax2,
#                       markersize=10)
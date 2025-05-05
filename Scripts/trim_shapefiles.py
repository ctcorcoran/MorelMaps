import os
import geopandas as gpd
import yaml
from shapely.geometry import box

# Load geo_config
geo_config = yaml.safe_load(open('../geo_config.yml'))


# Road and Land Shapefile Names
roads_raw_filename = '../Data/GIS/Raw/CA_FS_Roads/CA_FS_Roads.shp'
land_raw_filename = '../Data/GIS/Raw/Forest_Service_Land/Forest_Service_Land.shp'

roads_trimmed_folder = '../Data/GIS/FS Roads/'
land_trimmed_folder = '../Data/GIS/FS Land/'

roads_trimmed_filename = 'fs_roads.shp'
land_trimmed_filename = 'fs_land.shp'

# Load Perimeter Shapefile
perim_shp = gpd.read_file('../Data/GIS/Park Fire Perimeter 2024/Park_Fire_Perimeter_2024.shp').to_crs('EPSG:4326')

# Set Extent
extent_dict = {key:val[0] for key, val in perim_shp.bounds.to_dict().items()}
buffer = 0.1 # in deg

# extent = [lon_min, lon_max, lat_min, lat_max]
extent = [extent_dict[key] + buffer if key[0:3] == 'max' else extent_dict[key] - buffer for key in ['minx','maxx','miny','maxy']]

# Clipping Box - box(lon_min,lat_min,lon_max,lat_max)
extent_box = box(extent[0],extent[2],extent[1],extent[3])

# Write extent to geo_config
for key, val in {'lon_min':'minx','lon_max':'maxx','lat_min':'miny','lat_max':'maxy'}.items():
    geo_config['Extent'][key] = extent_dict[val]
    
with open('../geo_config.yml','w') as file:
    yaml.dump(geo_config,file)

# LOAD, CLIP, AND SAVE SHAPEFILES
# 'GeoDataFrame' object does not support the context manager protocol

# ROADS

fs_roads = gpd.read_file(roads_raw_filename)
fs_roads = fs_roads.clip(extent_box)
if not os.path.exists(roads_trimmed_folder):
    os.makedirs(roads_trimmed_folder)
fs_roads.to_file(roads_trimmed_folder + roads_trimmed_filename)
    
# LAND

fs_land = gpd.read_file(land_raw_filename)
fs_land = fs_land.clip(extent_box)
if not os.path.exists(land_trimmed_folder):
    os.makedirs(land_trimmed_folder)
fs_land.to_file(land_trimmed_folder + land_trimmed_filename)


import os
import yaml
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.merge import merge

import geopandas as gpd
from shapely.geometry import mapping

# Output Filepath
output_filepath = '../Data/GIS/dem.tif'

# Get Extent from geo_config
geo_config = yaml.safe_load(open('../geo_config.yml'))
extent_dict = geo_config['Extent']

# Get fire perimeter for masking the combined DEM
perim_filepath = '../Data/GIS/Park Fire Perimeter 2024/Park_Fire_Perimeter_2024.shp'
perim = gpd.read_file(perim_filepath).to_crs('EPSG:4326')

# Set Path for DEMs and load raw DEM tifs
dem_folder = '../Data/GIS/Raw/DEM/'

filenames = os.listdir(dem_folder)
dems = [rio.open(dem_folder+filename) for filename in filenames]

# Check if a temp_file exists - if it does, delete it:
if 'temp.tif' in os.listdir(dem_folder):
    os.remove(dem_folder+'temp.tif')

# Merge DEMs
mosaic_array, transform = merge(dems)

# Set metadata for new mosaic DEM from one of the constituent DEMs
mosaic_meta = dems[0].meta.copy()
mosaic_meta.update(
    {"driver": "GTiff",
        "height": mosaic_array.shape[1],
        "width": mosaic_array.shape[2],
        "transform": transform,
    }
)

# Temporarily write the mosaic DEM to file, so it can be opened again as a 
# readable (not writable) file and clipped (This is the only workflow I have
# found for clipping a combined DEM)

with rio.open(dem_folder+'temp.tif', "w", **mosaic_meta) as m:
    m.write(mosaic_array)

# Function to mask and clip the mosaic DEM
def prepare_raster(filename,perimeter):
    # read file
    with rio.open(filename,'r') as src:
        # Get clipping polygon, in DEM's crs
        geometry = perimeter.to_crs(src.crs).geometry.values[0]
        feature = [mapping(geometry)] # Required conversion
        
        # Clip, get array, transform
        src_clipped, src_out_transform = mask(src, feature, nodata=0, crop=True)
        src_crs = src.crs
    # Output crs is our standard
    out_crs = rio.CRS.from_string('EPSG:4326')
    
    # Reproject
    out,out_transform = rio.warp.reproject(src_clipped,
                          src_transform=src_out_transform,
                          src_crs = src_crs,
                          #dst_resolution = [1000,1000],
                          dst_crs=out_crs)
    return(out,out_transform,out_crs)

# Clip the DEM to the fire perimeter
dem_out, dem_trans, dem_crs = prepare_raster(dem_folder+'temp.tif',perim)

# Update metadata and write output to file for model use
mosaic_meta.update({
        "height": dem_out.shape[1],
        "width": dem_out.shape[2],
        "crs":str(dem_crs),
        "transform": dem_trans})

with rio.open(output_filepath, "w", **mosaic_meta) as m:
    m.write(dem_out)

# Again, check if a temp_file exists - if it does, delete it:
if 'temp.tif' in os.listdir(dem_folder):
    os.remove(dem_folder+'temp.tif')    

###################
# DIAGNOSTIC PLOTS

# # Plot DEMS to check merge success
# fig, ax = plt.subplots(1, figsize=(12, 12))
# show(mosaic_array, cmap='Greys_r', ax=ax)
# plt.axis("off")
# plt.show()

# # Plot to check clipping success
# fig2, ax2 = plt.subplots(1,1,figsize=(6,6))

# show(dem_out[0],#np.ma.masked_where(dem_out==0,dem_out),
#      transform=dem_trans,
#      ax=ax2,
#      alpha=1,
#      cmap='Greys')
    
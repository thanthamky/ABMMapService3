import numpy as np
import rasterio
from array_compressor import *
from agents_locator import *
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import os

class MapMaker:
    
    def __init__(self, base_raster_path):
        
        # Example usage:
        self.major_bbox = (99.0853175160000035,15.0483699459999993, 100.8600274380000030,16.2031785890000002)  # Full bounding box (left, bottom, right, top)
        self.full_width, self.full_height = 4726, 3144  # Full width and height in pixels
        self.pixel_size = (self.major_bbox[2] - self.major_bbox[0]) / self.full_width
        
        # Create the geospatial transform for the full array
        self.transform = from_origin(self.major_bbox[0], self.major_bbox[3], self.pixel_size, self.pixel_size)
        
        self.base_raster_path = base_raster_path
        
    def get_bbox_from_numpy_array(self, array, i, j, pixel_size, transform):
        """
        Get the bounding box for a given numpy array.

        Parameters:
        array (numpy.ndarray): The input numpy array.
        i (int): Starting row.
        j (int): Starting column.
        pixel_size (float): The size of each pixel.
        transform (rasterio.Affine): The geospatial transform for the full array.

        Returns:
        tuple: Bounding box coordinates (left, bottom, right, top) for the given array.
        """
        print(f'GET BBOX From NP i {i} j {j}')
        subset = array[i:, j:]
        height, width = subset.shape

        left = transform.c + j * transform.a
        top = transform.f + i * transform.e
        right = left + width * pixel_size
        bottom = top - height * pixel_size

        return (left, bottom, right, top)
    
    def subset_raster(self, raster_path, start_row, start_col, subset_width, subset_height, array_data):
        """
        Subset a raster using starting row and column indexes, and the width and height of the subset raster.

        Parameters:
        raster_path (str): The file path to the input raster.
        start_row (int): Starting row index.
        start_col (int): Starting column index.
        subset_width (int): Width of the subset raster.
        subset_height (int): Height of the subset raster.

        Returns:
        rasterio.io.DatasetReader: A rasterio DatasetReader object with the subset data.
        """
        # Read the input raster
        with rasterio.open(raster_path) as src:
            # Read the subset using window parameter
            
            print(f'subset raster start col {start_col} start row {start_row} width {subset_width} height {subset_height} ')
            
            window = rasterio.windows.Window(col_off=start_col, row_off=start_row,
                                             width=subset_width, height=subset_height)
            subset_data = src.read(1, window=window)

            # Create a new geospatial transform for the subset
            subset_transform = src.window_transform(window)

            # Create a new raster dataset for the subset
            profile = src.profile.copy()
            profile.update({"transform": subset_transform, "width": subset_width, "height": subset_height, "dtype": array_data.dtype})
            subset_raster = rasterio.io.MemoryFile().open(**profile)
            subset_raster.write(array_data, 1)

        return subset_raster
    
    def write_subset_raster(self, subset_raster_dataset, output_file_path, dst_crs='EPSG:4326'):
        """
        Write a raster dataset to a GeoTIFF file after reprojection, keeping the input shape.

        Parameters:
        subset_raster_dataset (rasterio.io.DatasetReader): The raster dataset to write.
        output_file_path (str): The file path to save the output GeoTIFF file.
        dst_crs (str, optional): The destination CRS (default: 'EPSG:4326').
        """
        src_crs = subset_raster_dataset.crs
        src_transform = subset_raster_dataset.transform
        src_width = subset_raster_dataset.width
        src_height = subset_raster_dataset.height

        # Calculate the new transform after reprojection
        dst_transform, _, _ = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src_width, src_height, *subset_raster_dataset.bounds)

        # Update the profile for the output raster
        output_profile = subset_raster_dataset.profile.copy()
        output_profile.update({"crs": dst_crs, "transform": dst_transform})

        # Read the source data
        src_data = subset_raster_dataset.read(1)

        # Create a destination numpy array to store the reprojected data
        dst_data = np.empty((src_height, src_width), dtype=src_data.dtype)

        # Perform the reprojection
        reproject(
            src_data,
            dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

        # Write the reprojected data to a GeoTIFF file
        with rasterio.open(output_file_path, 'w', **output_profile) as dst:
            dst.write(dst_data, 1)
            
        return output_file_path
            
    def make_map(self, map_data, start_row, start_col, sim_id, step, subset_width, subset_height, output_base_path):
    
        subset_bbox = self.get_bbox_from_numpy_array(map_data, start_row, start_col, self.pixel_size, self.transform)
        
        # Example usage:
        output_raster_path = os.path.join(output_base_path, f"{sim_id}_{str(step)}.tif")
        
        subset_raster_dataset = self.subset_raster(self.base_raster_path, start_row, start_col, subset_width, subset_height, map_data)

        saved_path = self.write_subset_raster(subset_raster_dataset, output_raster_path)
        
        return saved_path
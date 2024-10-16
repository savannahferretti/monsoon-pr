import warnings
import numpy as np
import xarray as xr
import dask
import dask.array as da
from datetime import datetime

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Purpose: A class to preprocess climate variables.
    Attributes:
        years (list): Years to filter the data.
        months (list): Months to filter the data.
        latrange (tuple): Latitude range for subsetting the region.
        lonrange (tuple): Longitude range for subsetting the region.
        levrange (tuple): Pressure level range for subsetting (if applicable).
        frequency (str): Time frequency to resample the data.
    """
    
    def __init__(self,years,months,latrange,lonrange,levrange,frequency):
        self.years     = years
        self.months    = months
        self.latrange  = latrange
        self.lonrange  = lonrange
        self.levrange  = levrange
        self.frequency = frequency

    @staticmethod
    def standardize_dims(data):
        """
        Purpose: Standardize/clean up the dimensions and coordinate data types.
        """
        dims = ['time','lat','lon','lev'] if 'lev' in data.dims else ['time','lat','lon']
        for dim in dims:
            if dim == 'time' and data.coords[dim].dtype.kind != 'M':
                data.coords[dim] = data.indexes[dim].to_datetimeindex()
            elif dim != 'time':
                data.coords[dim] = data.coords[dim].astype(float)
        return data.sortby(dims).transpose(*dims)

    def subset_dims(self,data):
        """
        Purpose: Subset the data by region (i.e., specified latitude/longitude range), time (i.e., specified years and months), 
                 and pressure level range (if applicable).
        """
        data = data.sel(lat=slice(*self.latrange),lon=slice(*self.lonrange))
        data = data.sel(time=(data['time.year'].isin(self.years))&(data['time.month'].isin(self.months)))
        if 'lev' in ds.dims:
            data = data.sel(lev=slice(*self.levrange))
        return data

    def resample_time(self,data):
        """
        Purpose: Resample the data to the specified time frequency.
        """
        data.coords['time'] = data.time.dt.floor(self.frequency)
        return data.groupby('time').first()

    def preprocess(self,data):
        """
        Purpose: Perform all preprocessing steps including standardizing dimensions,subsetting, and resampling time.
        """
        data = self.standardize_dims(data)
        data = self.subset_dims(data)
        if xr.infer_freq(data.time) != self.frequency:
            data = self.resample_time(data)
        return data

    @staticmethod
    def dataset(self,data,varname,longname,units,source,author,email):
        """
        Purpose: Convert the preprocessed data into an Xarray.Dataset and assign metadata attributes.
        """
        vardata = {varname:([*data.dims],data.data)}
        coords  = {dim: data.coords[dim].data for dim in data.dims}
        ds = xr.Dataset(vardata,coords)
        ds[varname].attrs = dict(long_name=longname,units=units)
        ds.time.attrs = dict(long_name='Time')
        ds.lat.attrs  = dict(long_name='Latitude', units='°N')
        ds.lon.attrs  = dict(long_name='Longitude', units='°E')
        if 'lev' in data.dims:
            ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
        ds.attrs = dict(source=source,history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
        return ds

    @staticmethod
    def save(data,savedir):
        """
        Purpose: Save the Xarray.Dataset to a netCDF file.
        """
        filename = f'OBS_{list(data.keys())[0]}.nc'
        data.to_netcdf(f'{savedir}/{filename}',compute=True)
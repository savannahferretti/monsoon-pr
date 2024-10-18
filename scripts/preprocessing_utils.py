import os
import warnings
import numpy as np
import xarray as xr
from datetime import datetime
import pystac_client as pystac
import planetary_computer
import fsspec

warnings.filterwarnings('ignore')

def get_era5():
    url  = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    data = xr.open_zarr(url, decode_times=True)
    data = data.rename({'latitude':'lat','longitude':'lon','level':'lev'})
    return data

def get_imerg():
    url     = 'https://planetarycomputer.microsoft.com/api/stac/v1'
    catalog = pystac.Client.open(url, modifier=planetary_computer.sign_inplace)
    assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
    data    = xr.open_zarr(fsspec.get_mapper(assets.href, **assets.extra_fields['xarray:storage_options']),consolidated=True)
    return data

def standardize_dims(data):
    dims = ['time','lat','lon','lev'] if 'lev' in data.dims else ['time','lat','lon']
    for dim in dims:
        if dim == 'time' and data.coords[dim].dtype.kind != 'M':
            data.coords[dim] = data.indexes[dim].to_datetimeindex()
        elif dim != 'time':
            data.coords[dim] = data.coords[dim].astype(float)
    return data.sortby(dims).transpose(*dims)

def subset_dims(data,years,months,latrange,lonrange,levrange=None):
    data = data.sel(lat=slice(*latrange),lon=slice(*lonrange))
    data = data.sel(time=(data['time.year'].isin(years))&(data['time.month'].isin(months)))
    if levrange and 'lev' in data.dims:
        data = data.sel(lev=slice(*levrange))
    return data

def resample_time(data,frequency):
    data.coords['time'] = data.time.dt.floor(frequency)
    return data.groupby('time').first()

def create_dataset(data,shortname,longname,units,source,author,email):
    vardata = {shortname:([*data.dims],data.data)}
    coords  = {dim:data.coords[dim].data for dim in data.dims}
    ds = xr.Dataset(vardata,coords)
    ds[shortname].attrs = dict(long_name=longname,units=units)
    ds.time.attrs = dict(long_name='Time')
    ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in data.dims:
        ds.lev.attrs = dict(long_name='Pressure level',units='hPa')
    date = datetime.today().strftime('%Y-%m-%d')
    ds.attrs = dict(source=source,history=f'Created on {date} by {author} ({email})')
    return ds

def save_dataset(data,savedir):
    filename = f'OBS_{list(data.keys())[0]}.nc'
    filepath = os.path.join(savedir,filename)
    data.to_netcdf(filepath)
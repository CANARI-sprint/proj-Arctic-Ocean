from scipy import signal
import xarray as xr
import numpy as np
import xmip.preprocessing as pp

##Create 4th order Bworth filter
def butter_lowpass(data,cut,order=4,sample_freq=1) :
    nyq = 0.5*sample_freq
    pass_freq =1./cut/nyq
    sos=signal.butter(order,pass_freq,'low',output='sos')
    filt=signal.sosfiltfilt(sos,data)
    return filt

def butter_ufunc(data,cut,tdim,order=4,sample_freq=1):
    filt = xr.apply_ufunc(
        butter_lowpass,
        data,
        cut,
        order,
        sample_freq,
        input_core_dims=[[tdim],[],[],[]],
        output_core_dims=[[tdim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=['float64']
        )
    return filt

def find_j_i(data, *, lat: float, lon: float):
        """
        A routine to find the nearest y x coordinates for a given latitude and longitude
        Usage: [y,x] = find_j_i(lat=49, lon=-12)

        :param lat: latitude
        :param lon: longitude
        :return: the y and x coordinates for the NEMO object's grid_ref, i.e. t,u,v,f,w.
        """
       # debug(f"Finding j,i for {lat},{lon} from {get_slug(data)}")

        dist2 = np.square(data.lat - lat) + np.square(data.lon - lon)
        [y, x] = np.unravel_index(np.argmin(dist2.data), dist2.shape)
        return [y, x]

def rename_cmip6(ds, rename_dict=None):  #Edited version of xmip preprocessing
    """Homogenizes cmip6 dataasets to common naming"""
    attrs = {k: v for k, v in ds.attrs.items()}
    ds_id = pp.cmip6_dataset_id(ds)

    if rename_dict is None:
        rename_dict = pp.cmip6_renaming_dict()

    # TODO: Be even stricter here and reset every variable except the one given in the attr
    # as variable_id
    # ds_reset = ds.reset_coords()

    def _maybe_rename_dims(da, rdict):
        for di in da.dims:
            for target, candidates in rdict.items():
                if di in candidates:
                    da = da.swap_dims({di: target})
                    if di in da.coords:
                        if not di==target:
                            da = da.rename({di: target}).set_xindex(target)
        return da

    # first take care of the dims and reconstruct a clean ds
    ds = xr.Dataset(
        {
            k: _maybe_rename_dims(ds[k], rename_dict)
            for k in list(ds.data_vars) + list(set(ds.coords) - set(ds.dims))
        }
    )

    rename_vars = list(set(ds.variables) - set(ds.dims))

    for target, candidates in rename_dict.items():
        if target not in ds:
            matching_candidates = [ca for ca in candidates if ca in rename_vars]
            if len(matching_candidates) > 0:
                if len(matching_candidates) > 1:
                    warnings.warn(
                        f"{ds_id}:While renaming to target `{target}`, more than one candidate was found {matching_candidates}. Renaming {matching_candidates[0]} to {target}. Please double check results."
                    )
                ds = ds.rename({matching_candidates[0]: target})

    # special treatment for 'lon'/'lat' if there is no 'x'/'y' after renaming process
    for di, co in [("x", "lon"), ("y", "lat")]:
        if di not in ds.dims and co in ds.dims:
            ds = ds.rename({co: di})

    # restore attributes
    ds.attrs = attrs
    return ds
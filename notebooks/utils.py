from scipy import signal
import xarray as xr

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

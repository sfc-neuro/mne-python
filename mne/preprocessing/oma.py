from ..utils import verbose, _check_preload
from ..io.pick import _picks_to_idx
from ..parallel import parallel_func

import numpy as np


@verbose
def fix_grad_artifact(raw, slices_per_volume, n_iter, n_cascades,
                      slice_duration='auto', TR='auto', picks='eeg', copy=True,
                      n_jobs=1, verbose=True):
    """Remove fMRI gradient artifact using OMA filter.

    Use the Optimized Moving Average algorithm to remove the gradient artifact
    from EEG data -- a very prominent artifact occuring during concurrent
    measures of EEG and (f)MRI data as detailed in
    :footcite:`FerreiraEtAl2016`.

    Parameters
    ----------
    raw : Raw
        The Raw EEG data we want to filter. The data need to be preloaded.
    slices_per_volume : int | None
        defaults to None
    n_iter : int
        The number of iterations of the filter. The more iteration, the
        tighter the filter. Defaults to [FIXME]
    n_cascades : int
        The number of cascades of the filter. Defaults to [FIXME]
    slice_duration : float | 'auto'
        Slice duration in seconds - default to Auto.
    TR : float | 'auto'
        Repetition time between two volumes in seconds - defaults to Auto. 
    %(picks_base)s all EEG channels.
    copy : bool
        Wether to make a copy of the data or operate in place.
        Defaults to True.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    raw_filt : Raw
        The filtered instance of the data.

    References
    ----------
    .. footbibliography::

    """
    _check_preload(raw, 'fix_grad_artifact')
    picks = _picks_to_idx(raw.info, picks, 'eeg', exclude=('bads'))
    if copy:
        raw = raw.copy()
    
    def OMA(M):
        # The filter is described in formula 12, 15 and 16 in the paper.
        # In order to transcribe these formulas directly into python code, we set up
        # the variables used in the formulas first.
        N = len(raw.times)
        k = np.arange(N)
        z = np.exp(1j * 2 * np.pi * k / N)

        J = n_iter
        L = n_cascades
        
        # Formula 12
        filt = (1 / M**2) * (1 - z**(-M)) * (1 - z**M) / ((1 - z**(-1)) * (1 - z))
        filt[z == 1+0j] = 0  # fix divide by zero cases
        
        # Formula 15
        filt = 1 - (1 - filt) ** J
        
        # Formula 16
        filt = filt ** L
        return filt
    
    filt = OMA(slice_duration)
    filt *= OMA(TR)

    def filt_channel(channel):
        """Apply the filter to a single channel."""
        signal = raw._data[channel]
        signal_fft = np.fft.fft(signal)
        signal_fft = filt * signal_fft
        signal = np.fft.ifft(signal_fft)
        raw._data[channel] = signal

    parallel, my_filt_func, _ = parallel_func(filt_channel, n_jobs,
                                              verbose=verbose)
    parallel(my_filt_func(channel) for channel in picks)

    return raw

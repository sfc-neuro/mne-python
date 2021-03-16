from ..utils import verbose, _check_preload
from ..io.pick import _picks_to_idx
import numpy as np
from ..io import RawArray

@verbose
def fix_grad_artifact(raw, slices_per_volume, n_iter,
                      slice_duration='auto', picks='eeg', copy=True,
                      verbose=True):
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
    slice_duration : float | 'auto'
        Slice duration in seconds - default to Auto.
    %(picks_base)s all EEG channels.
    copy : bool
        Wether to make a copy of the data or operate in place.
        Defaults to True.
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
    # Construct the values of z for which S was sampled
    N = len(raw.times)
    k = np.arange(N)
    z = np.exp(1j * 2 * np.pi * k / N)

    filt = (1 / (slice_duration**2) *
            (1 - z**(-slice_duration)) * (1 - z**slice_duration) /
            ((1 - z**(-1)) * (1 - z)))
    filt[z == 1+0j] = 0  # fix divide by zero cases
    filt = 1 - (1 - filt)**n_iter
    for channel in picks:
        signal = raw._data[channel]
        signal_fft = np.fft.fft(signal)
        signal_fft = filt * signal_fft

        signal = np.fft.ifft(signal_fft)
        raw._data[channel] = signal

    return raw

import mne

data_path = "/home/sfc/Documents/Aalto/GRA/lukeminen0058.vhdr"


raw = mne.io.read_raw_brainvision(data_path, eog=[ 'EOG1','EOG2', 'EOG3'], misc=['ECG'])

raw.crop(tmin=300, tmax=600)
raw.load_data()

clean_raw = mne.preprocessing.fix_grad_artifact(raw, slices_per_volume=21, n_iter= 100, n_cascades=1, slice_duration=308, TR=6468, picks='eeg', copy=True, n_jobs=1, verbose=True)



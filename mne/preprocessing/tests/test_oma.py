from mne.preprocessing import fix_grad_artifact
import mne

data_path = "/home/sfc/Documents/Aalto/GRA/lukeminen0058.vhdr"

raw = mne.io.read_raw_brainvision(data_path,
              eog=['Eog1', 'Eog2','Eog3', 'Ekg'], preload = True)
#raw.plot()
fix_grad_artifact(raw, 1,  1)

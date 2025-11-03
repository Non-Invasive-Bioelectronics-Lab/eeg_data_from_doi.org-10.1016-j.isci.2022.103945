# %%
# Intended to be run in a notebook environment

# %% Setup environment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
from scipy import signal

matplotlib.rcParams["pdf.fonttype"] = 42  # for vector fonts in plots
matplotlib.rcParams["ps.fonttype"] = 42  # for vector fonts in plots
plt.close("all")
import seaborn as sns

sns.set_theme()

%matplotlib ipympl


# %% Define functions

# Load data
def load_eeg(eeg_file):
    f = pyedflib.EdfReader(eeg_file)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    eeg = sigbufs.flatten()  # assumes only 1 channel recorded
    fs = f.getSampleFrequency(0)
    time = np.arange(eeg.shape[0]) / fs
    return eeg, time, fs


# %% Plot data

# AgAgCl references
agagcl_1, time_1, fs = load_eeg("agagcl_1_notch.edf")
plt.plot(time_1, agagcl_1)

agagcl_2, time_2, fs = load_eeg("agaglcl_2_notch.edf")
lag = 69.6
plt.plot(time_2 - lag, agagcl_2)

agagcl_3, time_3, fs = load_eeg("agagcl_3_notch.edf")
lag = -4.1
plt.plot(time_3 - lag, agagcl_3)


# One layer
mri_1_1, time_4, fs = load_eeg("MRI-1-20-1-1(3)_1_notch.edf")
lag = 2.4
plt.plot(time_4 - lag, -1 * mri_1_1) # note this record is inverted. Electrodes must have been connected the other way round, or IIRC camNtech has a setting for this

mri_1_2, time_5, fs = load_eeg("MRI-1-20-1-1(3)_2_notch.edf")
lag = -2.5
plt.plot(time_5 - lag, mri_1_2)

mri_1_3, time_6, fs = load_eeg("MRI-1-20-1-1(3)_3_notch.edf")
lag = 13.4
plt.plot(time_6 - lag, mri_1_3)

mri_1_4, time_7, fs = load_eeg("MRI-1-20-1-1(3)_4_notch.edf")
lag = 1.8
plt.plot(time_7 - lag, mri_1_4)


# Four layer
mri_4_1, time_8, fs = load_eeg("MRI-1-20-1-4(2)_1_notch.edf")
lag = -0.1
plt.plot(time_8 - lag, -1 * mri_4_1) # note this record is inverted. Electrodes must have been connected the other way round or IIRC camNtech has a setting for this

mri_4_2, time_9, fs = load_eeg("MRI-1-20-1-4(2)_2_notch.edf")
lag = 13.6
plt.plot(time_9 - lag, mri_4_2)

mri_4_3, time_10, fs = load_eeg("MRI-1-20-1-4(2)_3_notch.edf")
lag = 1.9
plt.plot(time_10 - lag, mri_4_3)

mri_4_4, time_11, fs = load_eeg("MRI-1-20-1-4(2)_4_notch.edf")
lag = -3.1
plt.plot(time_11 - lag, mri_4_4)


# Format plot
plt.xlabel("Time [s]")
plt.ylabel("EEG signal [uV])")

# %%

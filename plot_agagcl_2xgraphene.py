# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:47:14 2019

@author: mchi2cb3
"""
import pyedflib
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt, lfilter, welch
from scipy.fftpack import fft
from scipy.io import loadmat
import scipy.io as sio
from scipy import signal
import numpy as np
import time
import datetime
import matplotlib
import math
import seaborn as sns
import scipy.stats as stats

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set up the plotting style
# Set matplotlib to use constrained_layout as it formats figures better
plt.rcParams['figure.constrained_layout.use'] = True
# Ensure that we don't rasterise text in PDF output
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Set seaborn style to get the nice seaborn defaults
sns.set()
# Keep ticks on the bottom and left axes
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
matplotlib.rcParams['figure.dpi'] = 200


#first run the plotEDF.py script to import the EDF file

#agagcl_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2019-nazmul_graphene/DAQ_EEG_current_stim_agagcl_50hznotch.edf"
#graphene_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2019-nazmul_graphene/DAQ_EEG_current_stim_50hznotch.edf"



agagcl_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2021-nazmul-graphene-eeg/agagcl_3.edf"
graphene1_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2021-nazmul-graphene-eeg/MRI-1-20-1-1(3)_4.edf"
graphene4_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2021-nazmul-graphene-eeg/MRI-1-20-1-4(2)_4.edf"


#agagcl_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2019-nazmul_graphene/DAQ_EEG_voltagestim_atten80db_AgAgCl_50hznotch.edf"
#graphene_file = "//nasr.man.ac.uk/epsrss$/snapped/replicated/casson/data/small_projects/2019-nazmul_graphene/DAQ_EEG_voltagestim_atten80db_graphene_50hznotch.edf"

# Filter settings
fl = 47.5                                                                       # Low pass cut-off, Hz
fh = 52.5                                                                       # High pass cut-off, Hz
cutoffs = np.array([fl, fh])
order = 4                                                                       # Filter order, note that order is effectively twice this as using filtfilt        
filter_data = True


def filter_coefficients(cutoff, fs, order, filter_type):
    nyq = 0.5 * fs                                                              # Get nyquist frequency of signal
    normal_cutoff = cutoff / nyq                                                # Find the normalised cut-off frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)         # Generate array of filter co-efficients
    return b, a

# Filter data using filtfilt command
def data_filter(data, cutoff, fs, order):
    b, a = filter_coefficients(cutoff, fs, order, 'bandstop')                               # Call function to generate filter co-efficients
    signal_filtered = filtfilt(b, a, data)                                      # Perform filtering using filtfilt command
    return signal_filtered, b, a

def data_lowpass_filter(data, cutoff, fs, order):
    b, a = filter_coefficients(cutoff, fs, order, 'lowpass')
    signal_filtered = filtfilt(b, a, data)
    return signal_filtered


edf = pyedflib.EdfReader(agagcl_file)                                                  # create object 'edf' that holds file data
channels = edf.signals_in_file                                                  # find number of channels in file
fs = edf.getSampleFrequency(0)                                                  # get sample frequency of data. Assumes all channels have the same sample rate
T = 1/fs
agagcl = np.zeros((channels, edf.getNSamples()[0]))                             # generate empty array to store data
for i in np.arange(channels):                                                   # for each channel of data, store data from edf file in array
    agagcl[i, :] = edf.readSignal(i)
edf._close()                                                                    # close file to prevent an error next time code is run

graphene1_edf = pyedflib.EdfReader(graphene1_file)                                                  # create object 'edf' that holds file data
channels = graphene1_edf.signals_in_file                                                  # find number of channels in file
fs = edf.getSampleFrequency(0)                                                  # get sample frequency of data. Assumes all channels have the same sample rate
graphene1 = np.zeros((channels, graphene1_edf.getNSamples()[0]))                             # generate empty array to store data
for i in np.arange(channels):                                                   # for each channel of data, store data from edf file in array
    graphene1[i, :] = graphene1_edf.readSignal(i)
graphene1_edf._close()

graphene4_edf = pyedflib.EdfReader(graphene4_file)                                                  # create object 'edf' that holds file data
channels = graphene4_edf.signals_in_file                                                  # find number of channels in file
fs = edf.getSampleFrequency(0)                                                  # get sample frequency of data. Assumes all channels have the same sample rate
graphene4 = np.zeros((channels, graphene4_edf.getNSamples()[0]))                             # generate empty array to store data
for i in np.arange(channels):                                                   # for each channel of data, store data from edf file in array
    graphene4[i, :] = graphene4_edf.readSignal(i)
graphene4_edf._close()

if filter_data:
    agagcl_filtered, b, a = data_filter(agagcl.flatten(), cutoffs, fs, order)
    graphene1_filtered, __, __ = data_filter(graphene1.flatten(), cutoffs, fs, order)
    graphene4_filtered, __, __ = data_filter(graphene4.flatten(), cutoffs, fs, order)
    agagcl_filtered = data_lowpass_filter(agagcl_filtered, 50, fs, order)
    graphene1_filtered = data_lowpass_filter(graphene1_filtered, 50, fs, order)
    graphene4_filtered = data_lowpass_filter(graphene4_filtered, 50, fs, order)

agagcl = agagcl.flatten()
graphene1 = graphene1.flatten()
graphene4 = graphene4.flatten()
   


#agagcl_start_time = (int)(35.4121*fs)                                           # start time for old data agagcl, current stim
#graphene_start_time = (int)(19.1699*fs)                                         # start time for old data graphene, current stim

agagcl_start_time = int((15.0303*fs)+(15*fs))                                    # start time for new data agagcl, current stim
graphene4_start_time = int((20.2109*fs)+(1)+(15*fs))                              # start time for new data graphene, current stim

agagcl_start_time = 22105
graphene1_start_time = 28100
graphene4_start_time = 23100

#agagcl_start_time = (int)(13.8408*fs)                                           # start time for new data agagcl, voltage
#graphene_start_time = (int)(13.3525*fs)                                         # start time for new data graphene, voltage



agagcl_unfiltered_trimmed = agagcl[agagcl_start_time:agagcl_start_time+(20*60*1024)]
graphene1_unfiltered_trimmed = graphene1[graphene1_start_time:graphene1_start_time+(20*60*1024)]
graphene4_unfiltered_trimmed = graphene4[graphene4_start_time:graphene4_start_time+(20*60*1024)]
agagcl_filtered_trimmed = agagcl_filtered[agagcl_start_time:agagcl_start_time+(20*60*1024)]
graphene1_filtered_trimmed = graphene1_filtered[graphene1_start_time:graphene1_start_time+(20*60*1024)]
graphene4_filtered_trimmed = graphene4_filtered[graphene4_start_time:graphene4_start_time+(20*60*1024)]
data_time = np.arange(0,20*60,1/1024)
sleep_start_time = datetime.datetime(1989, 4, 25, 3, 3, 0)                      # time in original PSG record where playback started
my_datetimes = [sleep_start_time + datetime.timedelta(seconds=i) for i in data_time] # time array of datetimes for plotting


#plt.figure()
#plt.style.use('ggplot')
#plt.plot(data_time, agagcl_unfiltered_trimmed, label='Ag/AgCl Unfiltered')
#plt.plot(data_time, agagcl_filtered_trimmed, label='Ag/AgCl Filtered')
#plt.xlabel('Time / s')
#plt.ylabel('Voltage / μV')
#plt.legend()
#
#plt.figure()
#plt.plot(data_time, graphene_unfiltered_trimmed, label='Graphene Unfiltered')
#plt.plot(data_time, graphene_filtered_trimmed, label='Graphene Filtered')
#plt.xlabel('Time / s')
#plt.ylabel('Voltage / μV')
#plt.legend()
#
#plt.figure()
#plt.plot(data_time, agagcl_filtered_trimmed, label='Ag/AgCl Filtered')
#plt.plot(data_time, graphene_filtered_trimmed, label='Graphene Filtered')
#plt.xlabel('Time / s')
#plt.ylabel('Voltage / μV')
#plt.legend()

ax1 = plt.subplots(2, 1, figsize=(11,5.5))
ax1 = plt.subplot(2, 1, 1)
#plt.figure(num=1, figsize=(20, 4))
plt.plot(my_datetimes, graphene4_filtered_trimmed, label='Graphene Filtered')
plt.plot(my_datetimes, agagcl_filtered_trimmed, '--', label='Ag/AgCl Filtered')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-200, 200])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=6,seconds=30), my_datetimes[0]+datetime.timedelta(minutes=7, seconds=0)])


ax1 = plt.subplot(2, 1, 2)
plt.plot(my_datetimes, graphene4_filtered_trimmed, label='Textile Graphene Electrode')
plt.plot(my_datetimes, agagcl_filtered_trimmed, '--', label='Reference Ag/AgCl Electrode')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-200, 200])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=30)])
plt.legend(loc=4)

plt.subplots_adjust(hspace=0.50)
plt.tight_layout()
# plt.savefig("agagcl_vs_graphene.pdf", transparent=False)


# Make plot of all 3 types at same time
# Deep stage of sleep
ax1 = plt.subplots(3, 2, figsize=(11, 8))
ax1 = plt.subplot(3, 2, 1)
plt.plot(my_datetimes, agagcl_filtered_trimmed, label='Ag/AgCl Filtered', color='#66c2a5')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=6,seconds=30), my_datetimes[0]+datetime.timedelta(minutes=6, seconds=45)])


ax1 = plt.subplot(3, 2, 2)
plt.plot(my_datetimes, agagcl_filtered_trimmed, label='Ag/AgCl Filtered', color='#66c2a5')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19,seconds=20), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=35)])

ax1 = plt.subplot(3, 2, 3)
plt.plot(my_datetimes, graphene1_filtered_trimmed, label='Graphene 1 Filtered', color='#fc8d62')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=6,seconds=30), my_datetimes[0]+datetime.timedelta(minutes=6, seconds=45)])


ax1 = plt.subplot(3, 2, 4)
plt.plot(my_datetimes, graphene1_filtered_trimmed, label='Graphene 1 Filtered', color='#fc8d62')
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19,seconds=20), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=35)])

ax1 = plt.subplot(3, 2, 5)
plt.plot(my_datetimes, graphene4_filtered_trimmed, label='Graphene 4 Filtered', color='#8da0cb')
plt.legend()
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=6,seconds=30), my_datetimes[0]+datetime.timedelta(minutes=6, seconds=45)])


ax1 = plt.subplot(3, 2, 6)
plt.plot(my_datetimes, graphene4_filtered_trimmed, label='Graphene 4 Filtered', color='#8da0cb')
plt.legend()
plt.xlabel('Time of Day / HH:MM:SS')
plt.ylabel('Voltage / μV')
plt.ylim([-150, 150])
plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19,seconds=20), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=35)])

# plt.savefig('time_comparison.pdf')





# # Make plot of all 3 types at same time
# # Lighter stage of sleep
# ax1 = plt.subplots(3, 1, figsize=(11, 8))
# ax1 = plt.subplot(3, 1, 1)
# plt.plot(my_datetimes, agagcl_filtered_trimmed, label='Ag/AgCl Filtered', color='#66c2a5')
# plt.xlabel('Time of Day / HH:MM:SS')
# plt.ylabel('Voltage / μV')
# plt.ylim([-150, 150])
# plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=30)])

# ax1 = plt.subplot(3, 1, 2)
# plt.plot(my_datetimes, graphene1_filtered_trimmed, label='Graphene 1 Filtered', color='#fc8d62')
# plt.xlabel('Time of Day / HH:MM:SS')
# plt.ylabel('Voltage / μV')
# plt.ylim([-150, 150])
# plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=30)])

# ax1 = plt.subplot(3, 1, 3)
# plt.plot(my_datetimes, graphene4_filtered_trimmed, label='Graphene 4 Filtered', color='#8da0cb')
# plt.legend()
# plt.xlabel('Time of Day / HH:MM:SS')
# plt.ylabel('Voltage / μV')
# plt.ylim([-150, 150])
# plt.xlim([my_datetimes[0]+datetime.timedelta(minutes=19), my_datetimes[0]+datetime.timedelta(minutes=19, seconds=30)])




# Calculate FFT
N = int(len(agagcl_unfiltered_trimmed))
tf = np.linspace(0, int(1.0/(2.0*T)), int(N/2))
xf = fft(agagcl_unfiltered_trimmed)                                             # calculate fft
agagcl_unfiltered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

xf = fft(agagcl_filtered_trimmed)                                             # calculate fft
agagcl_filtered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

xf = fft(graphene1_unfiltered_trimmed)                                             # calculate fft
graphene1_unfiltered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

xf = fft(graphene1_filtered_trimmed)                                             # calculate fft
graphene1_filtered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

xf = fft(graphene4_unfiltered_trimmed)                                             # calculate fft
graphene4_unfiltered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

xf = fft(graphene4_filtered_trimmed)                                             # calculate fft
graphene4_filtered_trimmed_frequency = 2.0/N * np.abs(xf[:N//2])                 # convert two-sided complex values into single sided magnitude spectrum

f_graphene1, Pxx_den_graphene1 = welch(graphene1_unfiltered_trimmed, fs,  nperseg=30*fs, nfft=2**16)
f_graphene4, Pxx_den_graphene4 = welch(graphene4_unfiltered_trimmed, fs,  nperseg=30*fs, nfft=2**16)
f_agagcl, Pxx_den_agagcl = welch(agagcl_unfiltered_trimmed, fs,  nperseg=30*fs, nfft=2**16)
f_graphene1_filtered, Pxx_den_graphene1_filtered = welch(graphene1_filtered_trimmed, fs,  nperseg=30*fs, nfft=2**16)
f_graphene4_filtered, Pxx_den_graphene4_filtered = welch(graphene4_filtered_trimmed, fs,  nperseg=30*fs, nfft=2**16)



plt.figure(num=2, figsize=(11, 4))
plt.plot(tf, graphene4_unfiltered_trimmed_frequency, label='Textile Graphene Electrode')
plt.plot(tf, agagcl_unfiltered_trimmed_frequency, label='Reference Ag/AgCl Electrode')
plt.xlabel('Frequency / Hz')
plt.ylabel('Magnitude / μV')
plt.legend()
plt.ylim([0, 6])
plt.xlim([0, 512])
plt.tight_layout()
# plt.savefig("fft.pdf", transparent=False)

plt.figure(num=3, figsize=(11, 4))
plt.plot(f_graphene4, 10*np.log10(Pxx_den_graphene4), label='Textile Graphene Electrode')
plt.plot(f_agagcl, 10*np.log10(Pxx_den_agagcl), '--', label='Reference Ag/AgCl Electrode')
plt.xscale("log")
#plt.yscale("log")
plt.legend(loc=3)
plt.xlabel('Frequency / Hz')
#plt.ylabel(r'PSD / $\frac{μV^2}{Hz}$')
plt.ylabel(r'PSD / $\frac{dB}{Hz}$')
plt.tight_layout()


plt.figure(num=4, figsize=(11, 4))

plt.plot(f_agagcl, 10*np.log10(Pxx_den_agagcl), label='Reference Ag/AgCl Electrodes', color='#66c2a5')
plt.plot(f_graphene1, 10*np.log10(Pxx_den_graphene1), '--', label='Textile Graphene (1 Layer) Electrodes', color='#fc8d62')
plt.plot(f_graphene4, 10*np.log10(Pxx_den_graphene4), '-.', label='Textile Graphene (4 Layer) Electrodes', color='#8da0cb')
plt.xscale("log")
#plt.yscale("log")
plt.legend(loc=3)
plt.xlabel('Frequency / Hz')
#plt.ylabel(r'PSD / $\frac{μV^2}{Hz}$')
plt.ylabel(r'PSD / $\frac{dB}{Hz}$')
# plt.savefig('welch.pdf')
# plt.tight_layout()

This repository contains the EEG recordings carried out using a phantom, as used in paper https://doi.org/10.1016/j.isci.2022.103945 Fully printed and multifunctional graphene-based wearable e-textiles for personalized healthcare applications Islam, Md Rashedul et al. iScience, Volume 25, Issue 3, 103945.

These are stored in EDF format. See https://www.edfplus.info/ for info on this and common tools to load the data.

There are three different electrodes tested:

- Ag/AgCl ones as the reference. Files called agagcl_ ...
- One layer printed ones. Files called MRI-1-20-1-1 ...
- Four layer printed ones. Files called MRI-1-20-1-4 ...

Each one has repeat readings present, either 3 or 4.

Each recording is stored twice. One is the raw recording from the camNtech device, the other (with extension _notch) has the camNtech built in 50 Hz notching filtering applied. 

There is also a Python file main.py to plot all of these records and (approximately) align them in time. Note this is a new file generated now. 

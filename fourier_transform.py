import numpy as np # numerical calculations
import librosa # musical and audio sampling
#import matplotlib.pyplot as plt # make plots
import soundfile as sf # read and write audio
import scipy.fftpack as fft # Fourier tranform
from scipy.signal import medfilt # median filtering

input_audio = "sound.wav"
output_audio = "clean.wav"

# load audio
y, sampling_rate = librosa.load(input_audio, sr = None) # y = audio sign

# stft
# data in the frequency domain
S_full, phase = librosa.magphase(librosa.stft(y)) #magnitude + phase, SHORT TIME FOURIER TRANSFORM
#-> unpacks the sign into frequency domain ([from:to])
# S_full = magnitude (spectrum full)
# phase = phase
# x: time in segmented time windows
# y: frequency of each component

# noise average magnitude from 0.1sec
noise_power = np.mean(S_full[:, :int(sampling_rate*0.1)], axis=1)
# np.mean = average
# axis = 1 = calculating the average by coloumns
# -> for every row (frequency) we get the average magnitude for the first 0.1sec


# remove noise
# mask shows if the if the sign's magnitude is bigger than the noise level
mask = S_full > noise_power[:, None] # where exactly the audio is larger than the noise level
# distinguishes the audio and noise

# astype(float) makes float from the mask.
mask = mask.astype(float) # from numpy

# helps smoother the mask
mask = medfilt(mask, kernel_size=(1,5)) # median filter / középérték-szűrő
# kernel = ablak
# calculates a median
# 1 row x 5 coloumns
# only moves over the coloumns, and at every position takes the median of the actual 4 neighbors
# if there are only a few True values in a bigger False matrix, the median will be False (removing Trues)

# creates a cleaned spectrum cointaining the parts under the noise level
S_clean = S_full * mask
# True = 1.0 False = 0.0. so here we remove the NOISE.

# turn back to audio into time domain using ISTF (Inverse Short Time Fourier Transform)
y_clean = librosa.istft(S_clean * phase)
# x: time
# y: amplitude / magnitude

# write to a file
sf.write(output_audio, y_clean, sampling_rate)
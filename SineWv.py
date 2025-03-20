#   Created by: Saurabh Deshpande
#   Date: 03/01/2025
#   Description: Remove Noise from a Signal using FFT

from signal import Signals
import numpy as np
from matplotlib import pyplot as plt

smpl_rte=50000      #Hz
time=10             #sec
frq=500             #Hz

# 1 Define the sine function
def generate_sine_wave(freq,sample_rate,duration):
    x= np.linspace(0,duration,sample_rate*duration, endpoint=False)
    frequencies=x * freq
    
    y=np.sin((2*np.pi)* frequencies)
    return x,y
    
    
# 2 2 Sine waves - 1 with noise
_, normal_wave=generate_sine_wave(frq, smpl_rte, time)
_, noise_wve=generate_sine_wave(10000, smpl_rte, time)
noise_wve=noise_wve*0.3;
noise_tne=normal_wave+noise_wve


#Normalization of the results
nrmlz_tne=np.int16(noise_tne / noise_tne.max()*32767)
plt.subplot(2,1,1)
plt.plot(nrmlz_tne[: 1000])
plt.title('Original Noisy Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

# 3 FFT
from scipy.fft import rfft,rfftfreq, irfft
#No of samples in the normalized frequency
N=smpl_rte* time
print (N)
yf=rfft(nrmlz_tne)
xf=rfftfreq(N, 1/smpl_rte)

# 4 De-noising the Signal
# The maximum frequency is half the sample rate
pts_per_frq= len(xf) / (smpl_rte/2)
# Target Frequency is 10000 Hz
target_idx= int(pts_per_frq * 10000)
yf[target_idx -1 :target_idx+2]=0       # This gives 9k to 11k and will make it to 0

# Inverse FFT to get the cleaned time-domain signal
cleaned_signal = irfft(yf)

# Plot the cleaned time-domain signal
plt.subplot(2,1,2)
plt.plot(cleaned_signal[: 1000])
plt.title('Cleaned Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

# Show  Plot
plt.show()
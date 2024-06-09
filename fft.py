import numpy as np
import matplotlib.pyplot as plt
import math


def FFT(P):
    n = len(P)

    if n == 1:
        return P

    w = np.exp((2j * np.pi) / n)

    pEven = P[0::2]
    pOdd = P[1::2]

    yEven = FFT(pEven)
    yOdd = FFT(pOdd)

    y = [0] * n
    halfN = math.floor(n / 2)

    for k in range(halfN):
        y[k] = yEven[k] + pow(w, k) * yOdd[k]
        y[k + halfN] = yEven[k] - pow(w, k) * yOdd[k]

    return y


# Parameters
sampling_freq = 256
duration = 1.0
frequencies = [2, 3, 5]

t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
P = np.zeros(len(t))

for freq in frequencies:
    P += np.sin(2 * np.pi * freq * t)

fftresult = np.abs(FFT(P))
freqs = np.fft.fftfreq(len(fftresult), 1/sampling_freq)

for i in range(len(t)):
    fftresult[i] = fftresult[i] / (sampling_freq/2)

# Plot the time-domain signal
plt.subplot(2, 1, 1)
plt.plot(t, P)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Signal')

# Plot the frequency-domain representation
plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(fftresult))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency-Domain Signal (FFT)')
plt.xlim(0, 16)

plt.tight_layout()
plt.show()

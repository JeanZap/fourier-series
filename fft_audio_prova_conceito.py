import numpy as np
import librosa
from scipy.fftpack import fft, ifft
import soundfile as sf
import matplotlib.pyplot as plt

# Load an MP3 file
input_file = 'piano.mp3'
audio, sample_rate = librosa.load(input_file, sr=None)

# Extract 1 second of audio (adjust the start and end time as needed)
start_time = 0
end_time = start_time + 1
audio = audio[int(start_time * sample_rate):int(end_time * sample_rate)]

# Perform FFT
audio_fft = fft(audio)

for i in range(0, 17500):
    if(np.abs(audio_fft[i]) < 150):
        audio_fft[i] = 0

# for i in range(0, 3200):
#     if(audio_fft[i] < 120):
#         audio_fft[i] = 0

# for i in range(5000, 17500):
#     audio_fft[i] = 0

freqs = np.fft.fftfreq(len(audio_fft), 1 / sample_rate)
plt.plot(freqs, np.abs(audio_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency-Domain Signal (FFT)')
plt.xlim(0, sample_rate / 2)

plt.tight_layout()
plt.show()

# Manipulate the frequency domain data (e.g., remove some frequencies)
# Modify audio_fft as needed

# Perform Inverse FFT
audio_ifft = ifft(audio_fft)

# Save the modified audio to an MP3 file
output_file = 'audio_editado.mp3'
sf.write(output_file, audio_ifft.real, sample_rate)

print(f'Saved the processed audio to {output_file}')

import numpy as np
import matplotlib.pyplot as plt
import math
from pydub import AudioSegment
import ffmpeg

def FFT(P):
    n =  len(P)

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

# Specify the input MP3 file path
input_file = "piano.mp3"

# Define the start and end time for the one-second segment
start_time = 0  # Start at 0 seconds
end_time = 1000  # End at 1 second (1000 milliseconds)

# Read the one-second segment from the input MP3 file
audio = AudioSegment.from_file(input_file, format="mp3")

# Extract the one-second segment
audio_data = audio[start_time:end_time].get_array_of_samples()

fft = FFT(audio_data)
fftresult = np.abs(fft)

# for i in range(len(t)):
#     fftresult[i] = fftresult[i] / 256

# Calculate the time values for the x-axis
duration_in_seconds = len(audio_data) / audio.frame_rate
time_values = [t / len(audio_data) * duration_in_seconds for t in range(len(audio_data))]

inverseFft = np.fft.ifft(fft)

# Convert the complex array to 16-bit signed PCM audio data
real_part = np.abs(inverseFft)
normalized_audio_data = np.int16(real_part * (2 ** 15 - 1))

# Convert the audio data to bytes
audio_data_bytes = normalized_audio_data.tobytes()

# Create an AudioSegment object
audio = AudioSegment(
    data=audio_data_bytes,
    sample_width=2,  # 16-bit (2-byte) sample width
    frame_rate=audio.frame_rate,
    channels=1  # Mono audio
)

# Export the audio as an MP3 file
output_file = "audio_editado.mp3"
audio.export(output_file, format="mp3")

# Plot the time-domain signal
plt.subplot(2, 1, 1)
plt.plot(time_values, audio_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Signal')

sampling_freq = audio.frame_rate
freqs = np.fft.fftfreq(len(fftresult), 1/sampling_freq)

# Plot the frequency-domain representation
plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(fftresult))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency-Domain Signal (FFT)')
plt.xlim(0, sampling_freq / 2)

plt.tight_layout()
plt.show()
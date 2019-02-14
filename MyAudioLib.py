import pyaudio
import wave

import numpy
from matplotlib import pyplot, mlab
import scipy.io.wavfile
from PIL import Image
from collections import defaultdict

class Audio:
    def __init__(self):
        #Audio settings for record
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.RECORD_SECONDS = 10
        self.CHANNELS = 1
        self.RATE = 44100
        self.WAVE_OUTPUT_FILENAME = "output.wav"

        #Settings for creation Spectrogram
        self.SAMPLE_RATE = 8000  # Hz
        self.WINDOW_SIZE = 1024  # размер окна, в котором делается fft
        self.WINDOW_STEP = 256
        self.path_file = r'grad/grad-'


        #Spectrogram number
        self.count = 1

    #If a parameter RECORD_SECONDS is supplied to a function, its duration changes
    def record_audio_in_file(self, RECORD_SECONDS=None):

        self.RECORD_SECONDS = self.RECORD_SECONDS if RECORD_SECONDS is None else RECORD_SECONDS
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)+1):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def getting_wave_data(self, wave_filename):
        sample_rate, wave_data = scipy.io.wavfile.read(wave_filename)

        # Количество повторений
        number_repeat = int(len(wave_data) / sample_rate / 5)

        # assert sample_rate == self.SAMPLE_RATE, sample_rate

        if isinstance(wave_data[0], numpy.ndarray):  # стерео
            wave_data = wave_data.mean(1)

        return wave_data, number_repeat

    def show_spectrogram(self, wave_data, i):
        fig = pyplot.figure(figsize=(2, 2), frameon=False)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.specgram(wave_data,
                    NFFT=self.WINDOW_SIZE, noverlap=self.WINDOW_SIZE - self.WINDOW_STEP, Fs=self.SAMPLE_RATE)
        ax.axis('off')
        fig.savefig(((self.path_file+'{id}').format(id=i)), bbox_inches='tight')
        img = Image.open(self.path_file+'{id}.png'.format(id=i))
        area = (54, 14, 215, 175)
        cropped_img = img.crop(area)
        cropped_img.save(self.path_file+'{id}.png'.format(id=i))

    def creation_spectrogram(self):
        wave_filename = self.WAVE_OUTPUT_FILENAME
        wave_data, kol = self.getting_wave_data(str(wave_filename))
        if kol == 0:
            return
        step = int(len(wave_data) / kol)
        j = 0
        while j + step <= len(wave_data):
            self.show_spectrogram(wave_data[j:j + step], self.count)
            j += step
            self.count += 1



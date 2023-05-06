"""
Katya Borisova (boris040) and Nick Truong (truon351)
To run the program: python3 spectro.py nameOfWavFile.wav
example of running the program with sa1.wav: python3 spectro.py sa1.wav

Note: Imports matplotlib and numpy must be downloaded.
Note: Program needs to be run locally and not with SSH.
"""
import wave
import math 
import numpy.fft as np
import struct 
import matplotlib.pyplot as plt
import sys


def hamming(index, numSamples): # Applies hamming formula to given index based on number of samples and returns the value.
    if index < 0 or index >= numSamples:
        return 0
    product = 2*math.pi*index
    inside_cos = product/numSamples
    second_num = 0.46 * math.cos(inside_cos)
    return 0.54 - second_num


def spectro(wav):
    #Data Windows:
    wav = wave.open(wav, "r") # Reads .wav file
    samples_25ms = math.floor(16000 * (1/1000) * 25)
    samples_10ms = math.floor(16000 * (1/1000) * 10)
    window_25ms = [] # list of windows of data of size 25 ms. So list of lists of samples. Windows of data are overlapping.
    offset = 0 # Used to track the start position of every new window of data. Will be incremeted every 10 ms to help with overlapping the windows of data.
    while (offset + samples_25ms) < wav.getnframes(): #continue creating windows until the offset + window size reaches nframes
        wav.setpos(offset) # sets position to start of current window of data.
        window = []
        for i in range(samples_25ms): #add samples to the window one by one
            reading = struct.unpack("<h", wav.readframes(1))[0] #convert sample to an int
            product = reading * hamming(i, samples_25ms) #apply hamming function to the sample
            window.append(product)
        window_25ms.append(window)
        offset += samples_10ms #10 ms strides. So create new window of data every 10 ms which results in overlapping windows.

    #Fourier Transform:
    fft_25ms = [] # list of lists of frequency magnitudes at each timestep
    half = 0 # Only want information from the first half of array returned from fft
    for window in window_25ms: #loop through all the windows and apply the Fourier Transform
        fft = np.fft(window)
        pretty_fft = [] 
        half = len(fft)//2
        for Xk in fft[:half]: #only use the first half of fft's output and make it cleaner for the visualization
            real = Xk.real**2
            imaginary = Xk.imag **2
            square_magnitude = math.sqrt(real+imaginary)
            new_Xk = 10 * math.log(square_magnitude, 10)
            pretty_fft.append(new_Xk)
        fft_25ms.append(pretty_fft) 

    #Visualization:
    #creating every possible set of coordinates to create a grid
    x_list = [] #x_list = 0, 0, 0, 0, ...., 1, 1, 1 ....
    for i in range(0, len(fft_25ms)): 
        for j in range(0, half):
            x_list.append(i)
    y_list = [] #y_list = 1, 2, 3, 4, ...., 100, 101, ..., 1, 2, 3, ...
    for i in range(0, len(fft_25ms)):
        for j in range(0, half):
            y_list.append(j)

    #create plot
    plt.style.use('grayscale')
    c = plt.scatter(x_list, y_list, s = 0.12, c = fft_25ms, cmap='gray_r') # cmap='gray_r' handles inversion.
    plt.colorbar(c)
    plt.title("Spectrogram")
    plt.xlabel("Windows")
    plt.ylabel("Frequency")

    #display plot
    plt.show()


def main():
    """
    To run the program: python3 spectro.py nameOfWavFile.wav
    example of running the program with sa1.wav: python3 spectro.py sa1.wav
    """
    wave_file = sys.argv[1]
    spectro(wave_file)
      

if __name__ == '__main__':
   main()
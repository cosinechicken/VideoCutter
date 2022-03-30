from contextlib import closing
from PIL import Image, ImageFont, ImageDraw
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from numpy import dtype
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from datetime import datetime
import parselmouth
import matplotlib.pyplot as plt
import seaborn as sns

def printTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f = open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a")
    f.write(dt_string + ", ")
    print("CURRENT TIME: ", dt_string)

# Returns largest absolute value of any element in s (not necessarily 1D array)
def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

# Copy the frame and saves it
# (inputFrame, outputFrame): (int, int)
# Returns true if the source file exists, false otherwise.
def createNewFrame(outputFrame):
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    img = Image.new('RGB', (1280, 720), color = (255,255,255))
    fnt = ImageFont.truetype("C:\\Users\\brand\\Documents\\GitHub\\VideoCutter\\freemono.ttf", 30)
    frameText = "Volume: " + str(audioRMS[outputFrame])
    ImageDraw.Draw(img).text((0, 0), frameText, font=fnt, fill=(0, 0, 0))
    for i in range(400):
        if outputFrame > i and audioRMS[outputFrame-i] > -20 and audioRMS[outputFrame-i-1] > -20:
            ImageDraw.Draw(img).line((920-2*i, 600-4*audioRMS[outputFrame-i], 920-2*i-2, 600-4*audioRMS[outputFrame-i-1]), fill=(255,0,0,255), width=2)
    img.save(dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" frames saved.")

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    #assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:
        os.mkdir(s)
    except OSError:
        # assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"
        deletePath(s)
        try:
            os.mkdir(s)
        except OSError:
            assert False, "CreatePath failed twice in a row"

def deletePath(s): # Dangerous! Watch out!
    try:
        rmtree(s,ignore_errors=False)
    except OSError:
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

sns.set()

parser = argparse.ArgumentParser(description='Copies a video file.')
parser.add_argument('--input_file', type=str,  help='the video file you want modified')
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('--frame_rate', type=float, default=23.98, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=6, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 was the original default.")

args = parser.parse_args()

printTime()

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate

INPUT_FILE_NAME = args.input_file
INPUT_FILE = "C:\\Users\\brand\\Documents\\MIT_Course_Videos\\" + INPUT_FILE_NAME
FRAME_QUALITY = args.frame_quality
OUTPUT_FILE = INPUT_FILE
INPUT_FILE += ".mov"
OUTPUT_FILE += "-NEW.mp4"

# Path name: C:\Users\brand\TEMP
TEMP_FOLDER = "TEMP"
AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

createPath(TEMP_FOLDER)

# copy the audio into a wav file
command = "ffmpeg -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"

subprocess.call(command, shell=True)

# sampleRate: int, number of times per second the audio is sampled (44100)
# audioData: numpy array of int16's (116134912 samples, 2 channels)
sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
# Print data about sampleRate and audioData
print("AudioData shape: " + str(audioData.shape))   # (116134912, 2)
print("Datatype: " + str(dtype(audioData[0][0])))
###
audioSampleCount = audioData.shape[0]           # number of samples or "audio-frames" (116134912)
maxAudioVolume = getMaxVolume(audioData)        # largest amplitude in audioData (12189.0)
samplesPerFrame = sampleRate/frameRate          # number of times the audio is sampled per frame of the video at 30 frames per second (1470.0)
audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))      # Approximately number of frames in the video (79004)
###
print("AudioSampleCount: " + str(audioSampleCount))
print("MaxAudioVolume: " + str(maxAudioVolume))
print("SamplesPerFrame: " + str(samplesPerFrame))
print("AudioFrameCount: " + str(audioFrameCount))
###
audioRMS = np.zeros((audioFrameCount))

# Iterate through each video-frame
for i in range(audioFrameCount):
    start = int(i*samplesPerFrame)                              # Start audio-frame of video-frame i
    end = min(int((i+1)*samplesPerFrame),audioSampleCount)      # End audio-frame of video-frame i
    audiochunks = audioData[start:end].astype(float)            # Array of audio from start to end audio-frame of video-frame i
    temp = np.mean(np.square(audiochunks))
    audioRMS[i] = 20*np.log10(np.sqrt(temp))

lastExistingFrame = None

printTime()

for outputFrame in range(0, audioFrameCount):         # Iterate over all video-frames in this range
    createNewFrame(outputFrame)   # Copy the frame
    # This is necessary because the audio and video don't line up and we may run out of audio before we finish processing the video.

printTime()

wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,audioData)      # Combine the outputAudioData into a new .wav file at audioNew.wav

'''
outputFrame = math.ceil(outputPointer/samplesPerFrame)
for endGap in range(outputFrame,audioFrameCount):
    copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
'''
# Call the command which combines audioNew.wav with the video-frames of form newFrame%06d.jpg at 44100 frames per second, outputting to OUTPUT_FILE. %06d is integer with leading zeroes.
command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OUTPUT_FILE
subprocess.call(command, shell=True)

# Plot nice figures using Python's "standard" matplotlib library
snd = parselmouth.Sound(TEMP_FOLDER+"/audio.wav")
plt.figure()
plt.plot(snd.xs(), (snd.values.T))
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    print("Spectrogram drawn!")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")
    print("Intensity drawn!")

# intensity = snd.to_intensity()
# print("Intensity calculated!")
spectrogram = snd.to_spectrogram()
print("Spectrogram calculated!")
plt.figure()
draw_spectrogram(spectrogram)
# plt.twinx()
# draw_intensity(intensity)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

deletePath(TEMP_FOLDER) # Delete everything in TEMP_FOLDER
printTime()
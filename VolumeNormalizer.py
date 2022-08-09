from PIL import Image, ImageFont, ImageDraw
import subprocess
from numpy import dtype
from scipy.io import wavfile
import numpy as np
import pydub
import math
from shutil import rmtree
import os
import argparse
from datetime import datetime

def printTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a") as f:
        f.write(dt_string + ", ")
    print("CURRENT TIME: ", dt_string)

# Returns largest absolute value of any element in s (not necessarily 1D array)
def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def write(f, sr, x):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

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

parser = argparse.ArgumentParser(description='Copies a video file.')
parser.add_argument('--input_file', type=str,  help='the video file you want modified')
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('--frame_rate', type=float, default=23.98, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=6, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 was the original default.")
parser.add_argument('--end_volume', type=int, default=60, help="desired volume of video file in approximate decibels.")

args = parser.parse_args()

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate

for filename in os.listdir(args.input_file):
    with open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a") as f:
        f.write("VolumeNormalizer (" + filename + "): ")
    printTime()
    INPUT_FILE_NAME = filename[0:-4]
    INPUT_FILE = os.path.join(args.input_file, filename)
    FRAME_QUALITY = args.frame_quality
    END_VOLUME = args.end_volume

    # Path name: C:\Users\brand\TEMP
    TEMP_FOLDER = "TEMP_NORMALIZER"
    AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

    createPath(TEMP_FOLDER)

    # copy the audio into a wav file
    command = "ffmpeg -i \""+INPUT_FILE+"\" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"

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
        temp = np.maximum(np.mean(np.square(audiochunks)), 1)
        audioRMS[i] = (np.sqrt(temp))
        if i%1000 == 0:
            print(str(i) + " frames processed.")

    printTime()
    audioRMS = np.maximum(audioRMS, 40)
    volume = 20*np.log10(np.sqrt(np.mean(audioRMS**2)))
    mult = np.exp(np.log(10)*(END_VOLUME-volume)/20)
    # mult = 2
    print("Multiplier: " + str(mult))
    audioData = np.round(audioData * mult)
    newMaxVol = getMaxVolume(audioData)
    print("NEW MAX VOLUME: " + str(newMaxVol))
    if newMaxVol > 32768:
        print("ERROR: MAX VOLUME EXCEEDS 32768. ABORTING.")
        break

    write("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\output\\" + INPUT_FILE_NAME + ".mp3", SAMPLE_RATE, audioData)      # Combine the outputAudioData into a new .wav file at audioNew.wav
    printTime()

    with open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a") as f:
        f.write("\n")
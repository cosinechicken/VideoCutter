from contextlib import closing
from PIL import Image
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
from pytube import YouTube
from datetime import datetime

def printTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f = open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a")
    f.write(dt_string + ", ")
    print("CURRENT TIME: ", dt_string)
    f.close()

def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

# Returns largest absolute value of any element in s (not necessarily 1D array)
def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

# Copy the frame and saves it
# (inputFrame, outputFrame): (int, int)
# Returns true if the source file exists, false otherwise.
def copyFrame(inputFrame,outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

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

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input_file', type=str,  help='the video file you want modified')
parser.add_argument('--url', type=str, help='A youtube url to download and process')
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('--frame_rate', type=float, default=29.97, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=5, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 was the original default.")

args = parser.parse_args()

with open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a") as f:
    f.write("JumpCutter (" + args.input_file + "): ")

printTime()

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
########################## (1)
# if args.url != None:
#     INPUT_FILE = downloadFile(args.url)
# else:
#     INPUT_FILE = args.input_file
##########################

URL = args.url
FRAME_QUALITY = args.frame_quality
########################## (2)
# if len(args.output_file) >= 1:
#     OUTPUT_FILE = args.output_file
# else:
#     OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)
########################## (3)
INPUT_FILE_NAME = args.input_file
INPUT_FILE = "C:\\Users\\brand\\Documents\\MIT_Course_Videos\\" + INPUT_FILE_NAME
OUTPUT_FILE = INPUT_FILE
INPUT_FILE += ".mp4"
OUTPUT_FILE += "-NEW.mp4"
##########################
# Path name: C:\Users\brand\TEMP
TEMP_FOLDER = "TEMP"
AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

createPath(TEMP_FOLDER)

# split the video into individual frames
command = "ffmpeg -i \""+INPUT_FILE+"\" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner"
subprocess.call(command, shell=True)

# copy the audio into a wav file
command = "ffmpeg -i \""+INPUT_FILE+"\" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"

subprocess.call(command, shell=True)

# sampleRate: int, number of times per second the audio is sampled (44100)
# audioData: numpy array of int16's (116134912 samples, 2 channels)
sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
# Print data about sampleRate and audioData
print("AudioData shape: " + str(audioData.shape))   # (116134912, 2)
stringTemp = ""
# for i in range(100):
    # stringTemp += (str(audioData[10000000+i][0]) + " ")
print("StringTemp: " + stringTemp)
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
# Check each video-frame and see how loud the audio is in this frame
hasLoudAudio = np.zeros((audioFrameCount))

# Iterate through each video-frame
for i in range(audioFrameCount):
    start = int(i*samplesPerFrame)                              # Start audio-frame of video-frame i
    end = min(int((i+1)*samplesPerFrame),audioSampleCount)      # End audio-frame of video-frame i
    audiochunks = audioData[start:end]                          # Array of audio from start to end audio-frame of video-frame i
    maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume   # Ratio between the highest volume in audiochunks to highest volume in the video
    if maxchunksVolume >= SILENT_THRESHOLD:                             # Executes if the ratio above is large enough
        hasLoudAudio[i] = 1

# hasLoudAudio is 1 if the frame is loud enough and 0 otherwise
chunks = [[0,0,0]]          # 2-D array of following form: [[start-video-chunk, end-video-chunk, 0 if it shouldn't be included, 1 if the chunk should be included]]
shouldIncludeFrame = np.zeros((audioFrameCount))        # 0 if the video-frame shouldn't be included, 1 if the frame should be included.
for i in range(audioFrameCount):                        # Iterate through the video-frames
    # Consider only the frames within FRAME_SPREADAGE of i
    start = int(max(0,i-FRAME_SPREADAGE))               # Starting video-frame
    end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE)) # Ending video-frame
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end]) # 1 if anything in [start, end) has loud audio, 0 otherwise
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]):   # Did we flip? (e.g. shouldIncludeFrame is different between two consecutive frames)
        chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])        # Starting point of the chunk is the ending point of the previous chunk (chunks[-1][1]), and ending point of this chunk is just i

chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])  # Include the region from the last turning point to the end of the video
chunks = chunks[1:]     # [0, 0, 0], the first element, is just a filler element and should be removed

outputAudioData = np.zeros((0,audioData.shape[1]))
outputPointer = 0

lastExistingFrame = None

printTime()

for chunk in chunks:        # Iterate through each chunk (chunk is of form [start video-frame, end video-frame, 1 if included and 0 if not]
    audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)] # audioChunk is the array of sounds in audioData in the range of a given chunk
    
    sFile = TEMP_FOLDER+"/tempStart.wav"            # First temporary file
    eFile = TEMP_FOLDER+"/tempEnd.wav"              #
    wavfile.write(sFile,SAMPLE_RATE,audioChunk)     # Write the sounds in audioChunk to sFile
    # The below chunk of code takes sFile, runs it at the appropriate speed depending on if chunk[2] is 0 or 1, and outputs to eFile.
    with WavReader(sFile) as reader:
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
            tsm.run(reader, writer)
    ###
    _, alteredAudioData = wavfile.read(eFile)       # alteredAudiData is audioData for the frames in eFile, the shortened version of sFile

    leng = alteredAudioData.shape[0]                # Number of audio-frames in alteredAudioData
    endPointer = outputPointer+leng                 # endPointer is end position of outputAudioData
    outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume)) # Add alteredAudioData to outputAudioData after dividing by maxAudioVolume (so that all entries are doubles)

    #outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

    # smooth out transition's audio by quickly fading in/out
    
    if leng < AUDIO_FADE_ENVELOPE_SIZE:               # Executes if the cut version of this chunk is too short
        outputAudioData[outputPointer:endPointer] = 0 # audio is less than 400/44100 sec, let's just remove it.
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE  # premask = [0, 1/400, 2/400, ..., 399/400]
        mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo; mask = [[0, 0], [1/400, 1/400], ..., [399/400, 399/400]]. mask is the multiplier which masks audio-frames.
        outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask # fade in in 400/44100 seconds by multiplying each audio-frame by elements in mask
        outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask # same as above but fading out

    startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))    # Video-frame number of the starting frame in this chunk
    endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))         # Video-frame number of the ending frame in this chunk
    for outputFrame in range(startOutputFrame, endOutputFrame):         # Iterate over all video-frames in this range
        inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))  # inputFrame is the video-frame which should be considered that corresponds to the audio-frame outputFrame
        didItWork = copyFrame(inputFrame,outputFrame)   # Copy the frame, didItWork is true if the inputFrame file existed.
        # This is necessary because the audio and video don't line up and we may run out of audio before we finish processing the video.
        if didItWork:
            lastExistingFrame = inputFrame              # If the inputFrame file exists, then that becomes the last existing frame
        else:
            copyFrame(lastExistingFrame,outputFrame)    # If the inputFrame file doesn't exist, then resort to using the lastExistingFrame.

    outputPointer = endPointer  # Update the lower bound of the chunk (outputPointer)

printTime()
stringTemp = ""
# for i in range(50):
    # stringTemp += (str(outputAudioData[10000000+i][0]) + " ")
print("StringTemp: " + stringTemp)

wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)      # Combine the outputAudioData into a new .wav file at audioNew.wav

'''
outputFrame = math.ceil(outputPointer/samplesPerFrame)
for endGap in range(outputFrame,audioFrameCount):
    copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
'''
# Call the command which combines audioNew.wav with the video-frames of form newFrame%06d.jpg at 44100 frames per second, outputting to OUTPUT_FILE. %06d is integer with leading zeroes.
command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 \""+OUTPUT_FILE + "\""
subprocess.call(command, shell=True)


deletePath(TEMP_FOLDER) # Delete everything in TEMP_FOLDER
printTime()

with open("C:\\Users\\brand\\Documents\\MIT_Course_Videos\\Lecture_Cutter_Cmd.txt", "a") as f:
    f.write("\n")
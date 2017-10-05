import scipy
import scipy.io.wavfile as wav
import matplotlib.pylab as pylab
import numpy
from pydub import AudioSegment
import os
from shutil import copy2
import librosa
import librosa.display

AudioSegment.converter = r"D:\Program Files\ffmpeg\bin\ffmpeg.exe"

#to do update source code to remove issue with threads
#convert mp3 to .wav
def SaveMP3ToWAW(pathToMP3, pathToWAW):
    sound = AudioSegment.from_file(pathToMP3)
    sound.export(pathToWAW, format="wav")

#convert .mp3 to .wav intarget path
def ConvertToWavFiles(targetPath, extention = '.mp3'):
    mp3Files = GetListOfFilesByExt(tmpTarget, extention = extention)
    for item in mp3Files:
        tmpFile = os.path.splitext(item)[0]+".wav"
        SaveMP3ToWAW(item, tmpFile)

#helper function - delete all file in target dir by extension
def DeleteFilesByExt(targetDir, extention = '.png'):
    files = full_paths(targetDir)
    fileWithExtList = list()
    fileWithExtList += [each for each in files if each.endswith(extention)]
    for item in fileWithExtList:
        os.remove(item)

def GetListOfFilesByExt(targetDir, extention = '.png'):
    files = full_paths(targetDir)
    fileWithExtList = list()
    fileWithExtList += [each for each in files if each.endswith(extention)]
    return fileWithExtList

#helper function - count files by ext in dir
def CountFilesByExt(targetDir, extention = '.png'):
    files = full_paths(targetDir)
    fileWithExtList = list()
    fileWithExtList += [each for each in files if each.endswith(extention)]
    i = 0
    for item in fileWithExtList:
        i = i + 1
    return i

#convert wav to mel spectrogram
def WavToMelRosa(myAudio, saveTo):
    y, sr = librosa.load(myAudio)
    D = numpy.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256,
                                     fmax=10000)
    pylab.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S,
                                             ref=numpy.max),
                         y_axis='mel', fmax=10000,
                         x_axis='time')
    pylab.tight_layout()

    pylab.axis('off')
    pylab.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    pylab.savefig(saveTo + "_" + ".png",dpi = (200),  bbox_inches='tight', pad_inches=0.0)
    pylab.clf()
    pylab.close()

#get an wav file and create images for each second
def individualWavToSpectrogram(myAudio, saveTo):
    #print(myAudio)
    #Read file and get sampling freq [ usually 44100 Hz ]  and sound object
    samplingFreq, mySound = wav.read(myAudio)

    #Check if wave file is 16bit or 32 bit. 24bit is not supported
    mySoundDataType = mySound.dtype

    #We can convert our sound array to floating point values ranging from -1 to 1 as follows

    mySound = mySound / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel

    mySoundShape = mySound.shape
    samplePoints = float(mySound.shape[0])

    #Get duration of sound file
    signalDuration =  mySound.shape[0] / samplingFreq

    #If two channels, then select only one channel
    #mySoundOneChannel = mySound[:,0]

    #if one channel then index like a 1d array, if 2 channel index into 2 dimensional array
    if len(mySound.shape) > 1:
        mySoundOneChannel = mySound[:,0]
    else:
        mySoundOneChannel = mySound

    #Plotting the tone

    # We can represent sound by plotting the pressure values against time axis.
    #Create an array of sample point in one dimension
    timeArray = numpy.arange(0, samplePoints, 1)
    timeArray = timeArray / samplingFreq
    maxImages = int( round(signalDuration) )
    l = list()
    for i in range(0, maxImages) :
    #limit image to sound duration
        pylab.xlim((i * 1000, (i + 1) *  1000))
        
        #Scale to milliSeconds
        timeArray = timeArray * 1000

        pylab.rcParams['agg.path.chunksize'] = 100000

        #Plot the tone
        pylab.plot(timeArray, mySoundOneChannel, color='Black')
        #plt.xlabel('Time (ms)')
        #plt.ylabel('Amplitude')
        #print("trying to save")
        pylab.axis('off')
        pylab.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        pylab.savefig(saveTo + "_" + str(i) + ".png",dpi = (200),  bbox_inches='tight', pad_inches=0.0)
        #print("saved")
        l.append(pylab)
    for item in l:
        item.clf()
        item.close()
    
#returns list of paths from dir
def full_paths(targetPath):
    l = list()
    for d, dirs, files, in os.walk(targetPath):
        path = d
        for i in files:
            #print(path + '\\' + i)
            tmp = path + '\\' + i
            l.append(tmp)
    return l

#get sizes from list of files
def getSize(listOfFiles):
    size = list()
    for item in listOfFiles:
        tmp = os.path.getsize(item)
        size.append(tmp)
    return size

#convert .mp3 to .wav -> create spectrograms -> delete .wav
def createSpectroFromMP3(listOfSourceFiles):
    try:
        for item in listOfSourceFiles:
            tmpFile = os.path.splitext(item)[0]+".wav"
            SaveMP3ToWAW(item, tmpFile)
            base=os.path.basename(tmpFile)
            dirPath = os.path.dirname(item)
            #individualWavToSpectrogram(tmpFile, dirPath + '\\' + os.path.splitext(base)[0])
            WavToMelRosa(tmpFile, dirPath + '\\' + os.path.splitext(base)[0])
            os.remove(tmpFile)
    except Exception as ex:
        print(ex.args)
        pass

#helping function - get list of upper folders
def getUpperFolders(listOfFiles):
    l = list()
    for item in listOfFiles:
        tmp = os.path.split(item)[0]
        l.append(tmp)
    s = set()
    for i in l:
        tmp = i.split('\\')
        s.add(tmp[-1])
    return list(s)

#create training and test data
def createTrainTest(listOfFiles, targetPath, breakSample = 70 , extention = '.png'):
    pngResults = list()
    pngResults += [each for each in files if each.endswith(extention)]
    #get upper folders
    upperFolders = getUpperFolders(pngResults)

    #to do - create train and test folders with subfolders
    #and sample data

    #create test - train folders
    test = targetPath + "\\test"
    train =  targetPath + "\\train"
    if not os.path.exists(test):
        os.makedirs(test)
    if not os.path.exists(train):
        os.makedirs(train)

    for item in upperFolders:
        tmpItem = "\\" + item + "\\"
        #get all files in parent folder
        filteredByFolder = filter(lambda k: tmpItem in k, pngResults)
        filteredByFolderList = list(filteredByFolder)
        #get filenames
        file_names = list()
        for innerFile in filteredByFolderList:
            file_names.append(os.path.basename(innerFile))
        #split by "_" to get track/images numbers
        s = set()
        for innerName in file_names:
            s.add(innerName.split('_')[0])
        trackNumbers = list(s)
        numberTrain = round(len(trackNumbers)*breakSample/100)

        #create sub sub folder
        subTest = test + "\\" + item
        subTrain = train + "\\" + item
        if not os.path.exists(subTest):
            os.makedirs(subTest)
        if not os.path.exists(subTrain):
            os.makedirs(subTrain)
        
        trainList = trackNumbers[1:numberTrain]
        testList = trackNumbers[numberTrain:-1]
        
        #getTrain to copy files
        filteredTrain = list()

        for tr in trainList:
            filteredTrain.append( list( filter( lambda k: tr in k, filteredByFolderList ) ) )

        #getTest to copy files
        filteredTest = list()

        for te in testList:
            filteredTest.append( list( filter( lambda k: te in k, filteredByFolderList ) ) )

        flat_listTrain = [item for sublist in filteredTrain for item in sublist]
        flat_listTest = [item for sublist in filteredTest for item in sublist]

        #filteredTrainList = list(filteredTrain)
        #filteredTestList = list(filteredTest)
        
        #copy to train and test
        for i in flat_listTrain:
            copy2(i, subTrain)
        for m in flat_listTest:
            copy2(m, subTest)

pathToTestDataset = r"E:\NN_Music\fma_small\TEST_DATASET"
targetValidationPath = r"E:\NN_Music\fma_small\Ready for training"

trainPath = r'E:\NN_Music\fma_small\Ready for training\train'
testPath = r'E:\NN_Music\fma_small\Ready for training\test'

tmpTarget = r'E:\NN_Music\testing'
#ConvertToWavFiles(tmpTarget)



#files = full_paths(pathToTestDataset)
#mp3Files = GetListOfFilesByExt(tmpTarget, extention = '.mp3')



#createTrainTest(files, targetValidationPath)
#DeleteFilesByExt(pathToTestDataset)
#print( CountFilesByExt(testPath) )

#create spectrograms
#createSpectroFromMP3(files)


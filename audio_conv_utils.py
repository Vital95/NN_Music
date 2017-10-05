import numpy as np
from keras import backend as K
import scipy
import scipy.io.wavfile as wav
import matplotlib.pylab as pylab
import numpy
from pydub import AudioSegment
import os
from shutil import copy2
import librosa.display
import wawToImage as WTI
from PIL import Image

AudioSegment.converter = r"D:\Program Files\ffmpeg\bin\ffmpeg.exe"

def SaveMP3ToWAW(pathToMP3, pathToWAW):
    sound = AudioSegment.from_file(pathToMP3)
    sound.export(pathToWAW, format="wav")

TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']

def librosa_exists():
    try:
        __import__('librosa')
    except ImportError:
        return False
    else:
        return True

def preprocess_input(audio_path, dim_ordering='default'):
    '''Reads an audio file and outputs a Mel-spectrogram.
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if librosa_exists():
        import librosa
    else:
        raise RuntimeError('Librosa is required to process audio files.\n' +
                           'Install it via `pip install librosa` \nor visit ' +
                           'http://librosa.github.io/librosa/ for details.')

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    #convert .mp3 to .wav
    #tmpFile = os.path.splitext(audio_path)[0]+".wav"
    #SaveMP3ToWAW(audio_path, tmpFile)
    

    #src, sr = librosa.load(tmpFile, sr=SR)
    src, sr = librosa.load(audio_path, sr=SR)
    #os.remove(tmpFile)

    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[ int( (n_sample - n_sample_wanted) / 2 ):
                   int( (n_sample + n_sample_wanted) / 2 )]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MELS) ** 2,
              ref_power=1.0)

    if dim_ordering == 'th':
        x = np.expand_dims(x, axis=0)
    elif dim_ordering == 'tf':
        x = np.expand_dims(x, axis=3)
    return x


def decode_predictions(preds, top_n=5):
    '''Decode the output of a music tagger model.
    # Arguments
        preds: 2-dimensional numpy array
        top_n: integer in [0, 50], number of items to show
    '''
    assert len(preds.shape) == 2 and preds.shape[1] == 50
    results = []
    for pred in preds:
        result = zip(TAGS, pred)
        result = sorted(result, key=lambda x: x[1], reverse=True)
        results.append(result[:top_n])
    return results

#convert .wav to .png 
def preprocess_input_image(audio_path, dim_ordering='default'):
    '''Reads an audio file and outputs a Mel-spectrogram to image.
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if librosa_exists():
        import librosa
    else:
        raise RuntimeError('Librosa is required to process audio files.\n' +
                           'Install it via `pip install librosa` \nor visit ' +
                           'http://librosa.github.io/librosa/ for details.')

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(audio_path, sr=SR)

    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[ int( (n_sample - n_sample_wanted) / 2 ):
                   int( (n_sample + n_sample_wanted) / 2 )]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MELS) ** 2,
              ref_power=1.0)

    pylab.figure()
    librosa.display.specshow(x)
    pylab.tight_layout()

    base=os.path.basename(audio_path)
    dirPath = os.path.dirname(audio_path)
    saveTo = dirPath + '\\' + os.path.splitext(base)[0];
    pylab.axis('off')
    pylab.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    pylab.savefig(saveTo + "_" + ".png",dpi = (200),  bbox_inches='tight', pad_inches=0.0)
    pylab.clf()
    pylab.close()
    
    if dim_ordering == 'th':
        x = np.expand_dims(x, axis=0)
    elif dim_ordering == 'tf':
        x = np.expand_dims(x, axis=3)
    return x

#tmpTarget = r'E:\NN_Music\Good_Model_Music\testing acc\003'
#wavFiles = WTI.GetListOfFilesByExt(tmpTarget, extention = '.wav')
#for item in wavFiles:
#    preprocess_input_image(item) 

#compare image vs row data

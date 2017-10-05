from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from audio_conv_utils import decode_predictions, preprocess_input
import os
import wawToImage as WTI
import random
from random import randint

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5'
target = r'E:\NN_Music\testing\026_wrong_validate'


#if __name__ == '__main__':
#    model = MusicTaggerCRNN(weights='msd')
#    files = WTI.GetListOfFilesByExt(target, extention = '.wav')
#    thefile = open('top5.txt', 'w')
#    for item in files:
#        #audio_path = r'E:\NN_Music\Good_Model_Music\metal031040.mp3'
#        melgram = preprocess_input(item)
#        melgrams = np.expand_dims(melgram, axis=0)
#        preds = model.predict(melgrams)
#        #print(decode_predictions(preds,top_n=2))
#        thefile.write("%s\n" % item)
#        thefile.write("%s\n" % decode_predictions(preds,top_n=5))
#    thefile.close()




def getFeturesAndLabelsFromTarget(targetFolder):
    wavFiles = WTI.GetListOfFilesByExt(targetFolder, extention = '.wav')
    upperFolder = WTI.getUpperFolders(wavFiles)
    n = len(wavFiles)
    if len(set(upperFolder)) == 1:
        batch_features = np.zeros((n, 96, 1366, 1))
        batch_labels = np.zeros((n,1),dtype=np.int8)
        for i in range(0,n):
            batch_features[i] = preprocess_input(wavFiles[i])
            #correct
            # print(str(np.size(batch_features,0)))
            batch_labels[i] = 0
        return batch_features, batch_labels
    else:
        return None    

#generator for features
def MelGenerator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 96, 1366, 1))
    batch_labels = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
        # choose random index in features
            #to do review this line
            #index = random.choice(np.size(batch_features,1))
            index = randint(0, np.size(batch_features,0))
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

def getModel(input_tensor=None):
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    x = Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    return model

def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)

    if weights is None:
        return model
    else:
        # Load weights
        if K.image_dim_ordering() == 'tf':
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
        return model

#working good
#if __name__ == '__main__':
#    batch_features, batch_labels = getFeturesAndLabelsFromTarget(target)
#    model = getModel()
#    my_generator = MelGenerator(features = batch_features, labels = batch_labels, batch_size =  12)
#    model.compile(loss='binary_crossentropy',
#                optimizer='rmsprop',
#                metrics=['accuracy'])
#    model.fit_generator(my_generator, samples_per_epoch = 3, nb_epoch = 2, verbose=2,  callbacks=None, validation_data=None, class_weight=None, nb_worker=1)
#    model.save_weights('test012.h5')
#    print("done")


#testing binary model
if __name__ == '__main__':
     model = getModel()
     model.load_weights(r'E:\NN_Music\test012.h5')
     files = WTI.GetListOfFilesByExt(target, extention = '.wav')
     thefile = open('top1.txt', 'w')
     for item in files:
#        #audio_path = r'E:\NN_Music\Good_Model_Music\metal031040.mp3'
         melgram = preprocess_input(item)
         melgrams = np.expand_dims(melgram, axis=0)
         preds = model.predict(melgrams)
#        #print(decode_predictions(preds,top_n=2))
         thefile.write("%s\n" % item)
         thefile.write("%s\n" % preds)
     thefile.close()
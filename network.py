import numpy as np
from keras.models import Model
from keras.engine.topology import Input
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from imageio import imread
import skimage as sk
from skimage import transform
from skimage import util
from scipy import ndarray


def Network(inp_size, num_class):
    inp = Input(shape = inp_size)
    
    x = Conv2D(16, (3, 3), strides = 1, input_shape = inp_size, padding='valid')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(32, (3, 3), strides = (1,1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(48, (3, 3), strides = (1,1), padding='valid', bias_initializer = 'constant')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(64, (5, 5), strides = (1,1), bias_initializer = 'constant', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Flatten()(x)
    
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation = 'relu')(x)
    
    z = Dense(num_class, activation = 'softmax')(x)
    
    model = Model(inputs = inp, outputs= z)
    model.summary()
    return model


def generate_processed_batch(inp_data,label,img_size=(256,256,3),num_classes=14, batch_size = 32):
    img_rows, img_cols, channel = img_size
    batch_images = np.zeros((batch_size, img_rows, img_cols, channel))
    batch_label = np.zeros((batch_size, num_classes))
    while 1:
        for i_data in range(0,len(inp_data)-len(inp_data)%batch_size,batch_size):
            for i_batch in range(batch_size):
                img = imread(inp_data[i_data+i_batch])
                lab = np_utils.to_categorical(label[i_data+i_batch],num_classes)
                batch_images[i_batch] = img
                batch_label[i_batch] = lab
            yield batch_images, batch_label 
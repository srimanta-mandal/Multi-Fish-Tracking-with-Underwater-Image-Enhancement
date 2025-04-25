from __future__ import absolute_import
from __future__ import print_function
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

#code from https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
# def se_block(block_input, num_filters, ratio=8):                             # Squeeze and excitation block

# 	'''
# 		Args:
# 			block_input: input tensor to the squeeze and excitation block
# 			num_filters: no. of filters/channels in block_input
# 			ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced
			
# 		Returns:
# 			scale: scaled tensor after getting multiplied by new channel weights
# 	'''

# 	pool1 = GlobalAveragePooling2D()(block_input)
# 	flat = Reshape((1, 1, num_filters))(pool1)
# 	dense1 = Dense(num_filters//ratio, activation='relu')(flat)
# 	dense2 = Dense(num_filters, activation='sigmoid')(dense1)
# 	scale = multiply([block_input, dense2])
	
# 	return scale


import numpy as np

import random
import os
from cv2 import imread,resize,INTER_AREA
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Lambda,LSTM,BatchNormalization,LeakyReLU,PReLU,Add,Concatenate
from keras import Sequential
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop,Adam
from keras import initializers, regularizers, optimizers
from keras import backend as K
from keras.regularizers import l2
from keras.initializers import VarianceScaling
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import numpy.random as rng

import tensorflow as tf


def contrastive_loss(y_true, y_pred):
    margin = 0.6
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def W_init(shape,name=None):
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def b_init(shape,dtype=None):
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,dtype=dtype) #,name=name)


def build_siamese_model(input_shape):
    main_input = Input(input_shape)
    #uwcnn layers
    
    layer1 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(main_input)
    # print("1", layer1.shape)
    
    # print("1.1", layer1.shape)
    layer2 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer1)
    layer3 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer2)
    concat_1 = Concatenate(axis = 3)([layer1,layer2,layer3,main_input])
    lay1 = BatchNormalization()(concat_1)
    layer4 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(lay1)
    layer5 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer4)
    layer6 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer5)
    concat_2 = Concatenate(axis = 3)([concat_1,layer4,layer5,layer6])
    lay2 = BatchNormalization()(concat_2)
    layer7 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(lay2)
    layer8 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer7)
    layer9 = Conv2D(16,(3,3),activation = 'relu',strides=(1,1),padding="same")(layer8)
    # print("9", layer9.shape)
    concat_3 = Concatenate(axis = 3)([concat_2,layer7,layer8,layer9])
    lay3 = BatchNormalization()(concat_3)
    layer10 = Conv2D(3,(3,3),activation = 'relu',strides=(1,1),padding="same")(lay3)
    layer10 = BatchNormalization()(layer10)
    # print("10", layer10.shape)
    out = Add()([main_input,layer10])
    # print("out", out.shape)


    # Squeeze and excitation layers
    se = squeeze_excite_block(out, ratio = 3)
    # se = se_block(out, 53,8)
    #out = Add()([main_input, se])


    #fully   connected   layers
    flat= Flatten()(se)

    ld1 = Dense(4096,activation='relu')(flat)

    b1 = BatchNormalization()(ld1)
    
    ld2 = Dense(1024,activation='relu')(b1)
    
    b2 = BatchNormalization()(ld2)

    ld3 = Dense(512,activation='relu')(b2)
    
    output = BatchNormalization()(ld3)

    model = Model(main_input, output)

    return model


def SiameseNetwork(input_shape):
    top_input = Input(input_shape)

    bottom_input = Input(input_shape)
    featureExtractor = build_siamese_model(input_shape)
    encoded_top = featureExtractor(top_input)
    encoded_bottom = featureExtractor(bottom_input)
    

    
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    L1_distance = L1_layer([encoded_top, encoded_bottom])

    prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)

    siamesenet = Model(inputs=[top_input,bottom_input],outputs=prediction)

    return siamesenet



# import the necessary packages

import numpy as np
import argparse
import cv2

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



def loadimgs(path,n = 0):
    X=[]
    y = []
    curr_y = n
    
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        alphabet_path = os.path.join(path,alphabet)
        
        category_images=[]

        for filename in os.listdir(alphabet_path):
          if(len(category_images)>=16):
            break

          image_path = os.path.join(alphabet_path, filename)
          print(image_path)
          # image = imread(image_path).astype('float32')/255
          image = imread(image_path)
          #print(image.shape)


          img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
          # cv2_imshow(img_yuv)
          # equalize the histogram of the Y channel
          clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
          # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
          img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
          # cv2_imshow(img_yuv)
          # convert the YUV image back to RGB format
          image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
          # cv2_imshow(img_output)
          # cv2.imwrite("fish_eqh.png", img_output)


          # Gamma Correction
          output = adjust_gamma((255 * image).astype(np.uint8), gamma = 0.5)
          image = (output.astype(np.float)) / 255
          image = resize(image,(121,53),interpolation = INTER_AREA)
          image = np.array(image)
          image = image.astype('float32')/255
          category_images.append(image)
          y.append(curr_y)
          
        try:
          X.append(np.stack(category_images))
        except ValueError as e:
            print(e)
            #print('Hi')
            #print("error - category_images:", category_images)
        curr_y += 1
    # print("shape y", len(y))
    # print("shape X", len(X),X[0].shape,X[1].shape)
    y = np.vstack(y)
    X = np.stack(X)
    return X,y

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    num_classes = 23
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    # print("value of n", n)
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            #print('HI')
            # each folder should have same number of image ex 1447 here
            f21 = z1//16
            l31 = z1 % 16
            f22 = z2//16
            l32 = z2 % 16
            pairs += [[x[f21][l31], x[f22][l32]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            f21 = z1//16
            l31 = z1 % 16
            f22 = z2//16
            l32 = z2 % 16
            pairs += [[x[f21][l31], x[f22][l32]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels).astype("float32")


# path to dataset
data_path = '/content/drive/MyDrive/BTPData_new/fish_image'


X,y = loadimgs(data_path)
# print(y.shape)
digit_indices = [np.where(y == i)[0] for i in range(23)]
# print(len(digit_indices))
tr_pairs,tr_y = create_pairs(X,digit_indices)
# print(tr_pairs.shape)

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

# visualize(tr_pairs[:-1], tr_y[:-1], to_show=4, num_col=4)
input_shape = (53,121,3)
model = SiameseNetwork(input_shape)
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rms = RMSprop()
# print(model.summary())
# model.compile(loss='mse', optimizer=rms, metrics=['acc'])


keras_callbacks   = [
      EarlyStopping(monitor='loss', patience=4, mode='min', min_delta=0.0001),
      # ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
]
rms = RMSprop()
print(model.summary())
model.compile(loss='mse', optimizer=rms, metrics=['acc'])


history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y.astype('float32'),
          batch_size=32,
          epochs=30,
          validation_split = 0.25,
          callbacks = keras_callbacks
          )

save_path = "/content/drive/MyDrive/Multi-Fish-Tracking-with-Underwater-Image-Enhancement/models/model30_f4k_ft.h5"
model.save(save_path)

figure, axis = plt.subplots(1, 2)
# Plot training & validation accuracy values
axis[0].plot(history.history['acc'])
axis[0].plot(history.history['val_acc'])
axis[0].set_title('Model accuracy')
axis[0].set_ylabel('Accuracy')
axis[0].set_xlabel('Epoch')
axis[0].legend(['Train', 'Test'], loc='upper left')
plt.savefig('Train_val_accuracy.png')
plt.show()

# Plot training & validation loss values
axis[1].plot(history.history['loss'])
axis[1].plot(history.history['val_loss'])
axis[1].set_title('Model loss')
axis[1].set_ylabel('Loss')
axis[1].set_xlabel('Epoch')
axis[1].legend(['Train', 'Test'], loc='upper left')
plt.savefig('Siamese_plotaugment.png')
plt.show()


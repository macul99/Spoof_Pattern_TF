from spoof_net import SpoofDenseNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pickle
import argparse
#import cv2
import os
from os import listdir, makedirs
from os.path import join, exists
import keras
from keras.callbacks import TensorBoard
from keras.models import load_model
import time
import tensorflow as tf
import math


class PrepareData():
    def __init__(self, data_source, feature_list):
        'Initialization'
        self.data_source = data_source        
        self.feature_list = feature_list
    def get_data(self):
        data = []
        labels = []
        for file in self.data_source:            
            with open(file,'rb') as f:
                input_dic = pd.read_pickle(f)
                input_ft = input_dic['feature']
                class_label = input_dic['class']
                for index, row in input_ft.iterrows():
                    print(index) 
                    #if index>5000:
                    #    break           
                    ft = []
                    for d in self.feature_list:
                        ft += list(row[d])
                    if np.NaN not in ft:
                        data += [ft]
                        labels += [class_label]
        return np.array(data), np.array(labels)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, dim, n_classes=3, batch_size=32, shuffle=True):
        'Initialization'
        self.data = data        
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.data[indexes], keras.utils.to_categorical(self.labels[indexes], num_classes=self.n_classes)
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def arcface_loss(embedding, labels, out_num=3, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output

params = {  'batch_size': 32,
            'shuffle': True,
            'test_size': 0.1,
            'random_state': 42,
            'learning_rate': 1e-5,
            'learning_decay': 1e-4,
            'reg_coef': 0.0005,
            'metrics': ['accuracy'],
            'data_source': [    '/media/macul/black/spoof_db/feature_extraction/real_1.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/real_3.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/image_attack_2.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/image_attack_4.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/video_attack_2.pkl',
                                ],
            'val_source': [     #'/media/macul/black/spoof_db/feature_extraction/real_2.pkl',
                                #'/media/macul/black/spoof_db/feature_extraction/real_4.pkl',
                                #'/media/macul/black/spoof_db/feature_extraction/real_5.pkl',
                                #'/media/macul/black/spoof_db/feature_extraction/real_6.pkl',
                                #'/media/macul/black/spoof_db/feature_extraction/image_attack_1.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/image_attack_3.pkl',
                                #'/media/macul/black/spoof_db/feature_extraction/video_attack_1.pkl',
                                ],
            'feature_list': ['CoALBP_GREY',]#'LPQ', 'BSIF','LBP_RG','COLOR_MOMENT',],
          }

'''
data_dic = {'data_source': [    '/media/macul/black/spoof_db/feature_extraction/real_1.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/real_3.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/image_attack_2.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/image_attack_4.pkl',
                                '/media/macul/black/spoof_db/feature_extraction/video_attack_2.pkl',
                                ],
            'feature_list': ['LPQ','BSIF','LBP_GREY','CoALBP', 'CoALBP_GREY', CoALBP_256','LBP_RG','COLOR_MOMENT'],
            }
'''


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log", required=True,
    help="path to log directory")
ap.add_argument("-e", "--epochs", type=int, default=50,
    help="# of epochs to train our network for")
args = vars(ap.parse_args())

if not exists(args['log']):
    makedirs(args['log'])

tensorboard = TensorBoard(log_dir=args['log'], batch_size=params['batch_size'])

LABELS = set(["Real", "Image_Attack", "Video_Attack"])

data_prepare = PrepareData(params['data_source'], params['feature_list'])
data, labels = data_prepare.get_data()
val_data_prepare = PrepareData(params['val_source'], params['feature_list'])
val_data, val_labels = val_data_prepare.get_data()

(trainX, testX, trainY, testY) = train_test_split(  data, labels,
                                                    test_size=params['test_size'], 
                                                    stratify=labels, 
                                                    random_state=params['random_state'])

opt = Adam(lr=params['learning_rate'], decay=params['learning_decay'] / args["epochs"])
model = SpoofDenseNet.build(feature_size=trainX.shape[1],
                            classes=len(LABELS), 
                            reg=l2(params['reg_coef']))
'''
model.compile(  loss="categorical_crossentropy", 
                optimizer=opt,
                metrics=params['metrics'])
'''
model.compile(  loss=arcface_loss, 
                optimizer=opt,
                metrics=params['metrics'])

#training_generator = DataGenerator(trainX, trainY, trainX.shape[1], batch_size=params['batch_size'])
#validation_generator = DataGenerator(testX, testY, testX.shape[1], batch_size=params['batch_size'])
training_generator = DataGenerator(data, labels, data.shape[1], batch_size=params['batch_size'])
validation_generator = DataGenerator(val_data, val_labels, val_data.shape[1], batch_size=params['batch_size'])

H = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=training_generator.__len__(),
                        validation_steps=validation_generator.__len__(),
                        epochs=args['epochs'],
                        callbacks=[tensorboard])

model.save(join(args['log'],'model.h5'))

N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(join(args["log"],'plot.png'))
plt.show()
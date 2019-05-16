from os import path
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

BASE_PATH = './dataset' # soft link to '/media/macul/black/spoof_db/train_test_all/tfrecord'
#BASE_PATH = './dataset1' # soft link to '/media/macul/black/spoof_db/train_test_all/tfrecord1'
#BASE_PATH = '/media/macul/black/spoof_db/train_test_all/tfrecord'

CFG_PATH = './config'

#DATASET_MEAN = path.sep.join([BASE_PATH, "mean.json"])

TRAIN_REC = path.sep.join([BASE_PATH, "train5-00000-of-00005"])
VAL_REC = path.sep.join([BASE_PATH, "validation5-00000-of-00005"])

NUM_CLASSES = 3 

OUT_PATH = './output'
PREFIX = 'train_1'
NET_SCOPE = 'SpoofDenseNet'

# feature parameter
Feature_Type = ['COLOR_MOMENT_S3'] # LBP_GREY, LBP_RG, CoALBP, CoALBP_256, CoALBP_GREY, BSIF, LPQ, COLOR_MOMENT, COLOR_MOMENT2

Feature_Size = {'BSIF'              : 256,
                'CoALBP'            : 6144,
                'CoALBP_GREY'       : 2048,
                'CoALBP_256'        : 512,
                'COLOR_MOMENT'      : 36,
                'COLOR_MOMENT_S3'   : 81,
                'COLOR_MOMENT2'     : 27,
                'COLOR_MOMENT3'     : 72,
                'LBP_GREY'          : 822,
                'LBP_RG'            : 522,
                'LPQ'               : 256,
               }

# model parameter
Embedding_size = 16
Dense_Stem_Cfg = [Embedding_size,]
Dense_Cfg = {'BSIF'            : [128, 64, 32],
             'CoALBP'          : [512, 128, 32],
             'CoALBP_GREY'     : [32, 32],
             'CoALBP_256'      : [32,],
             'COLOR_MOMENT'    : [16,],
             'COLOR_MOMENT_S3' : [16,],
             'COLOR_MOMENT2'   : [16,],
             'COLOR_MOMENT3'   : [16,],
             'LBP_GREY'        : [32, 32],
             'LBP_RG'          : [32, 32],
             'LPQ'             : [64, 32],
            }
Drop_Rate = {'BSIF'            : 0.75,
             'CoALBP'          : 0.75,
             'CoALBP_GREY'     : 0.75,
             'CoALBP_256'      : 0.9,
             'COLOR_MOMENT'    : 0.9,
             'COLOR_MOMENT_S3' : 0.9,
             'COLOR_MOMENT2'   : 0.9,
             'COLOR_MOMENT3'   : 0.9,
             'LBP_GREY'        : 0.75,
             'LBP_RG'          : 0.9,
             'LPQ'             : 0.75,
            }
Regularizer = l2(5e-4)
              
Initializer = tf.contrib.layers.xavier_initializer(uniform=False)
Activation = tf.nn.relu



# argumentation parameter
Img_shuffle_grid_size = 0 # set to <=1 to disable

# optimizer parameter
#Opt_name = 'SGD' # SGD, Adam
Opt_lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
Opt_lr_steps = [300000, 600000, 900000, 1200000]
Opt_momentum = 0.9
#Opt_weight_decay = 0.0005
#Opt_rescale_grad = 1.0

# loss type
LOSS_TYPE = 'softmax' # 'arc' or 'softmax'

# loss parameter
Arc_margin_angle = 0.0 # arc face margin_m
Arc_margin_scale = 64.0 # arc face margin_s

# training parameters
BUFFER_SIZE = 5000
BATCH_SIZE = 128
DEVICE_IDS = "0,"
NUM_DEVICES = len(DEVICE_IDS.split(","))
NUM_EPOCH = 10
Update_Interval = 10


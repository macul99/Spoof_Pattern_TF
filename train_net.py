import tensorflow as tf
import numpy as np
import argparse
import logging
import json
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile
from config import net_config as config
from SpoofDspTf import SpoofDspTf

# set gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_IDS

# python -m train_net --checkpoints /home/macul/libraries/mk_utils/mx_facerecog_resnet50/output --prefix firstTry
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
#ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=-1, help="epoch to restart training at")
args = vars(ap.parse_args())


# create SpoofDspTf(config)
SpoofVal = SpoofDspTf(config)

# backup net_config file
copyfile(config.CFG_PATH+'/net_config.py', os.path.sep.join([SpoofVal.checkpointsPath,'net_config_{}.py'.format(args["start_epoch"]+1)]))
copyfile(__file__, os.path.sep.join([SpoofVal.checkpointsPath,os.path.basename(__file__).split('.')[0]+'_{}'.format(args["start_epoch"]+1)+'.py']))

# start training
SpoofVal.train_net(args["start_epoch"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d
#from sklearn.cluster import KMeans
#from scipy.interpolate import Rbf, interp2d
#from scipy import fftpack, ndimage
from scipy.optimize import minimize
#from tfFaceDet import tfFaceDet
import time
import argparse
import pickle
from imutils import paths
from spoofing_lbp.SpoofDspTf import SpoofDspTf

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--ckpt-num", required=True, help="checkpoint number")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-o", "--output-path", required=True, help="output folder path")
args = vars(ap.parse_args())

ckpt_num = args['ckpt_num']
prefix = args['prefix']
import sys
sys.path.append('/home/macul/libraries/mk_utils/tf_spoof/output/{}'.format(prefix))
import net_config_0 as config
config.BASE_PATH = '/media/macul/black/spoof_db/train_test_all/tfrecord'
config.CFG_PATH = '/home/macul/libraries/mk_utils/tf_spoof/config'
config.OUT_PATH = '/home/macul/libraries/mk_utils/tf_spoof/output'
config.PREFIX = prefix

SpoofVal = SpoofDspTf(config)

sess, embeddings, logit, pred, acc, features_tensor = SpoofVal.deploy_net(ckpt_num)

#faceDet = tfFaceDet()
#f_log = open('/home/macul/Projects/realsense/distance_face_size.log','wb')
'''
input_dic = {   'real'          : [ 
                                    '/media/macul/black/spoof_db/video_shotbyego/see',
                                    '/media/macul/black/spoof_db/zhaoguang_video_sz/group0_output6',
                                    '/media/macul/black/spoof_db/zhaoguang_video_sz/group3_output2'
                                  ], 
                'image_attack'  : [ 
                                    '/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/test_cropped',
                                    '/media/macul/black/spoof_db/spoofing_data_Mar_2019/picture_crop_test'
                                  ], 
                'video_attack'  : [ 
                                    '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/screen_attack',
                                    '/media/macul/black/spoof_db/spoofing_data_Mar_2019/video_crop_test' 
                                  ]
            }


input_dic = {   'real'          : [ 
                                    '/media/macul/black/spoof_db/train_test_all/train1/real',
                                    '/media/macul/black/spoof_db/train_test_all/test1/real',
                                    '/media/macul/black/spoof_db/train_test_all/test/real',
                                  ], 
                'image_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/train1/image_attack',
                                    '/media/macul/black/spoof_db/train_test_all/test1/image_attack',
                                    '/media/macul/black/spoof_db/train_test_all/test/image_attack',
                                  ], 
                'video_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/train1/video_attack',
                                    '/media/macul/black/spoof_db/train_test_all/test1/video_attack',
                                    '/media/macul/black/spoof_db/train_test_all/test/video_attack',
                                  ]
            }
'''

# for COLOR_MOMENT_S3 only
input_dic = {   'real'          : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/real',
                                  ], 
                'image_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/image_attack',
                                  ], 
                'video_attack'  : [ 
                                    '/media/macul/black/spoof_db/train_test_all/test5/video_attack',
                                  ]
            }
output_dic = {}

time_elaps = []

for key in input_dic.keys():
    fd_list = input_dic[key]
    tmp_dic = {}
    for fd in fd_list:
        tmp_rst = []
        imagePaths = list(paths.list_images(fd))
        count = 0
        for imgP in imagePaths:
            count += 1            
            frame = cv2.imread(imgP)
            print(key, fd, count, imgP, frame.shape) 
            [h, w] = frame.shape[:2]
            start_time = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            #print('crop_flag: ', crop_flag)            
            rst = SpoofVal.eval(frame, sess, pred, features_tensor)
            time_elaps += [time.time() - start_time]
            tmp_rst += [np.squeeze(rst[0])]
        tmp_dic[fd] = np.array(tmp_rst)
    output_dic[key] = tmp_dic


with open(args['output_path']+'/{}_{}.pkl'.format(prefix, ckpt_num),'wb') as f:
    pickle.dump(output_dic, f)

print(np.mean(time_elaps))
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

#f_log.close()
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
import cPickle as pkl
from dan import dan
from ssd import ssd
import time
import os
from skimage import feature
from scipy.signal import convolve2d
from scipy.io import loadmat
from scipy.stats import skew
from os import listdir, makedirs
from os.path import join, exists
import pickle
import pandas as pd
from imutils import paths
from spoof_feature import LBP_GREY, LBP_RG, CoALBP, CoALBP_256, CoALBP_GREY, BSIF, LPQ, COLOR_MOMENT

def round_int(x):
    return int(round(x))

def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def similar_transform_matrix(src_points, dst_points):
    num_points = src_points.shape[0]
    #print('num_points: ',num_points)
    X = np.zeros([num_points*2, 4]).astype(np.float64)
    U = np.zeros([num_points*2, 1]).astype(np.float64)
    for i in range(num_points):
        U[i, 0] = src_points[i,0]
        U[i+num_points, 0] = src_points[i,1]
        X[i, 0] = dst_points[i,0]
        X[i, 1] = dst_points[i,1]
        X[i, 2] = 1.0
        X[i, 3] = 0.0
        X[i+num_points, 0] = dst_points[i,1]
        X[i+num_points, 1] = -dst_points[i,0]
        X[i+num_points, 2] = 0.0
        X[i+num_points, 3] = 1.0
    X_t = np.transpose(X)
    XX = np.matmul(X_t, X)
    r = np.matmul(np.matmul(np.linalg.inv(XX), X_t), U)
    Tinv = np.zeros([3,3])
    Tinv[0,0] = r[0]
    Tinv[0,1] = -r[1]
    Tinv[0,2] = 0.0
    Tinv[1,0] = r[1]
    Tinv[1,1] = r[0]
    Tinv[1,2] = 0.0
    Tinv[2,0] = r[2]
    Tinv[2,1] = r[3]
    Tinv[2,2] = 1.0
    Tinv = np.transpose(Tinv)
    T = np.linalg.inv(Tinv)
    return T[0:2,:]

# find the landmarks and the associated area
def landmark_area(fun_p, img, bbox=None):
    points, score = fun_p(img, num_points=68, boundingBox=bbox)
    points_5 = np.zeros([5,2])
    points_5[0,:] = [np.mean(points[36:42,0]), np.mean(points[36:42,1])]
    points_5[1,:] = [np.mean(points[42:48,0]), np.mean(points[42:48,1])]
    points_5[2,:] = [np.mean(points[30,0]), np.mean(points[30,1])]
    points_5[3,:] = [np.mean(points[48,0]), np.mean(points[48,1])]
    points_5[4,:] = [np.mean(points[54,0]), np.mean(points[54,1])]
    area = np.zeros([10,4]).astype(np.uint8) # [x1,y1,x2,y2] for reye, leye, nose, mouth, r_eyebrow, l_eyebrow, face, btw eyes, btw eyebrows, nose tip
    area[0,:] = [int(np.min(points[36:42,0])), int(np.min(points[36:42,1])), int(np.max(points[36:42,0]))+1, int(np.max(points[36:42,1]))+1]
    area[1,:] = [int(np.min(points[42:48,0])), int(np.min(points[42:48,1])), int(np.max(points[42:48,0]))+1, int(np.max(points[42:48,1]))+1]
    area[2,:] = [int(np.min(points[27:36,0])), int(np.min(points[27:36,1])), int(np.max(points[27:36,0]))+1, int(np.max(points[27:36,1]))+1]
    area[3,:] = [int(np.min(points[48:68,0])), int(np.min(points[48:68,1])), int(np.max(points[48:68,0]))+1, int(np.max(points[48:68,1]))+1]
    area[4,:] = [int(np.min(points[17:22,0])), int(np.min(points[17:22,1])), int(np.max(points[17:22,0]))+1, int(np.max(points[17:22,1]))+1]
    area[5,:] = [int(np.min(points[22:27,0])), int(np.min(points[22:27,1])), int(np.max(points[22:27,0]))+1, int(np.max(points[22:27,1]))+1]
    area[6,:] = [int(np.min(points[0:17,0])),  int(np.min(points[0:17,1])),  int(np.max(points[0:17,0]))+1,  int(np.max(points[0:17,1]))+1]
    area[7,:] = [int(np.max(points[36:42,0])), int(np.min(points[36:48,1])),int(np.min(points[42:48,0]))+1,  int(np.max(points[36:48,1]))+1]
    area[8,:] = [int(np.max(points[17:22,0])), int(np.max(points[17:27,1])-4),int(np.min(points[22:27,0]))+1,  int(np.max(points[17:27,1])+4)+1]
    area[9,:] = [int(points[30,0]+0.5)-1, int(points[30,1]+0.5)-1, int(points[30,0]+0.5)+2,  int(points[30,1]+0.5)+2]
    return points, points_5, area


# convert list of different row length to numpy array
'''
v = [[1], [1, 2]]
import itertools
x=np.array(list(itertools.zip_longest(*v, fillvalue=0))).T # for python3
x=np.array(list(itertools.izip_longest(*v, fillvalue=0))).T # for python2
'''

# class 0 - real, 1 - image_attack, 2 - video_attack
folder_dic = {  '/media/macul/black/spoof_db/collect_iim/traintest/train_iim_cropped'           : {'class': 0, 'dataset': 'real_1'}, # 7641
                '/media/macul/black/spoof_db/collect_iim/traintest/see'                         : {'class': 0, 'dataset': 'real_2'}, # 795
                '/media/macul/black/spoof_db/iim/iim_real_jul_2018_cropped'                     : {'class': 0, 'dataset': 'real_3'}, # 12386
                '/media/macul/black/spoof_db/iim/toukong_real_jul_2018_cropped'                 : {'class': 0, 'dataset': 'real_4'}, # 6783, test
                '/media/macul/black/spoof_db/video_shotbyego/see'                               : {'class': 0, 'dataset': 'real_5'}, # 4107
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/positive/iim'   : {'class': 0, 'dataset': 'real_6'}, # 59
                '/media/macul/black/spoof_db/iim/iim_image_attack_cropped'                      : {'class': 1, 'dataset': 'image_attack_1'}, # 271
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/image_attack'   : {'class': 1, 'dataset': 'image_attack_2'}, # 445
                '/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/test_cropped'   : {'class': 1, 'dataset': 'image_attack_3'}, # 2142
                '/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/train_cropped'  : {'class': 1, 'dataset': 'image_attack_4'}, # 11282
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/screen_attack'  : {'class': 2, 'dataset': 'video_attack_1'}, # 1069
                '/media/macul/black/spoof_db/collect_fferr/iim_screrr/traintest/see'            : {'class': 2, 'dataset': 'video_attack_2'}, # 10124
             }

feature_list = {'LPQ':LPQ(), 
                'BSIF':BSIF(), 
                'LBP_GREY': LBP_GREY(),
                'CoALBP': CoALBP(),
                'CoALBP_GREY': CoALBP_GREY(),
                'CoALBP_256': CoALBP_256(),
                'LBP_RG': LBP_RG(),
                'COLOR_MOMENT': COLOR_MOMENT(),
                }

rst_dic_folder = '/media/macul/black/spoof_db/feature_extraction'
#rst_dic_file = '/media/macul/black/spoof_db/feature_extraction/rst_dic.pkl'

for k, v in folder_dic.iteritems():
    print(k, v)
    if exists(join(rst_dic_folder,v['dataset']+'.pkl')):
        with open(join(rst_dic_folder,v['dataset']+'.pkl'),'rb') as f:
            rst_dic = pickle.load(f)
    else:
        rst_dic = v.copy()
        with open(join(rst_dic_folder,v['dataset']+'.pkl'),'wb') as f:
            pickle.dump(rst_dic, f, pickle.HIGHEST_PROTOCOL)

    if 'feature' in rst_dic.keys():
        df_feature = rst_dic['feature']
        columns_df = list(df_feature.columns)
        for ft in feature_list.keys():
            if ft not in columns_df:
                model = feature_list[ft]
                df_feature[ft] = pd.Series(np.NaN, index=df_feature.index)
                for im_name in list(df_feature['img_name']):
                    print(ft,im_name)
                    img = cv2.imread(join(k, im_name))
                    #df_feature[ft][df_feature['img_name']==im_name] = pd.Series([model(img)])     
                    df_feature.loc[df_feature['img_name']==im_name, ft] = pd.Series([model(img)],index=df_feature[df_feature['img_name']==im_name].index) 
                    #print(df_feature.loc[df_feature['img_name']==im_name, ft])              
    else:
        df_feature = pd.DataFrame(columns=['img_name']+feature_list.keys())
        imagePaths = list(paths.list_images(k))
        col = feature_list.keys()
        for imPath in imagePaths:
            im_name = imPath.split('/')[-1]
            img = cv2.imread(imPath)
            value = []
            for ft in col:
                print(ft,im_name)
                value += [feature_list[ft](img)]                
            df_feature = df_feature.append(pd.Series(value+[im_name],col+['img_name']), ignore_index=True)
        rst_dic['feature'] = df_feature

    #print(rst_dic['feature'])
    with open(join(rst_dic_folder,v['dataset']+'.pkl'),'wb') as f:
        pickle.dump(rst_dic, f, pickle.HIGHEST_PROTOCOL)

'''
import numpy as np
import pandas as pd
import pickle as pkl
f=open('/media/macul/black/spoof_db/feature_extraction/rst_dic.pkl','rb')
rst_dic=pkl.load(f)
rst_dic['/media/macul/black/spoof_db/collect_iim/traintest/train_iim_cropped']['feature']
rst_dic['/media/macul/black/spoof_db/collect_iim/traintest/train_iim_cropped']['feature']['LPQ'].values[0]
'''
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
from tfFaceDet import tfFaceDet
from tfMtcnnFaceDet import tfMtcnnFaceDet
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
import argparse
from SpoofDspTf import SpoofDspTf

def prepare_img(img, bbox, crop_scale_to_bbox=2.5, crop_square=True):
    img_h, img_w, _ = img.shape
    bbx_x, bbx_y, x2, y2 = bbox
    bbx_w = x2 - bbx_x + 1
    bbx_h = y2 - bbx_y + 1
    
    #print('orig_img_shape: ', img.shape)

    crp_w = int(bbx_w*crop_scale_to_bbox)
    crp_h = int(bbx_h*crop_scale_to_bbox)

    if crop_square:
        crp_w = max(crp_w, crp_h)
        crp_h = crp_w

    img_crop = np.zeros([crp_h, crp_w, 3]).astype(np.uint8)
    img_crop[:,:,1] = 255 # make empty portion green to differentiate with black

    
    crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
    crp_x2 = min(int(bbx_x-(crp_w-bbx_w)/2.0)+crp_w-1, img_w-1)
    crp_y2 = min(int(bbx_y-(crp_h-bbx_h)/2.0)+crp_h-1, img_h-1)

    delta_x1 = -min(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    delta_y1 = -min(0, int(bbx_y-(crp_h-bbx_h)/2.0))

    img_crop[delta_y1:delta_y1+crp_y2-crp_y1+1, delta_x1:delta_x1+crp_x2-crp_x1+1] = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:].copy()

    bbx_x1 = bbx_x - crp_x1 + delta_x1
    bbx_y1 = bbx_y - crp_y1 + delta_y1
    bbx_x2 = bbx_x1 + bbx_w -1
    bbx_y2 = bbx_y1 + bbx_h -1

    #img_crop=cv2.rectangle(img_crop, (bbx_x1, bbx_y1), (bbx_x2, bbx_y2), (0,255,0), 3)
    
    return img_crop, [bbx_x1, bbx_y1, bbx_x2, bbx_y2]

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--ckpt-num", required=True, help="checkpoint number")
ap.add_argument("-mf", "--model-folder", required=True, help="model folder path")
#ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-v", "--video-path", required=True, help="input video full path")
ap.add_argument("-t", "--threshold", required=True, type=float, help="threshold")
ap.add_argument("-tf", "--threshold-false", required=True, type=float, help="threshold for false detection")
ap.add_argument("-s", "--min-size", required=False, type=int, default=100, help="threshold")
args = vars(ap.parse_args())

args['model_folder'] = os.path.abspath(args['model_folder'])

prefix = args['model_folder'][args['model_folder'].rfind('/')+1:]
out_folder = args['model_folder'][0:args['model_folder'].rfind('/')]

import sys
sys.path.append(args['model_folder'])
import net_config_0 as config
config.OUT_PATH = out_folder

ckpt_num = args['ckpt_num']

SpoofVal = SpoofDspTf(config)

sess, embeddings, logit, pred, acc, features_tensor = SpoofVal.deploy_net(ckpt_num)


useMtcnn = True
if useMtcnn:
    faceDet = tfMtcnnFaceDet()
else:
    faceDet = tfFaceDet()

#f_log = open('/home/macul/Projects/realsense/distance_face_size.log','wb')

#cap = cv2.VideoCapture('/home/macul/test1.avi')
#cap = cv2.VideoCapture('/media/macul/black/spoof_db/record_2018_08_13_17_31_27.avi')
#cap = cv2.VideoCapture('/home/macul/iim_sz_02_05_mov.avi')
#cap = cv2.VideoCapture('/media/macul/black/spoof_db/spoofing_data_Mar_2019/picture/record_2019_02_21_10_03_03.avi')
#cap = cv2.VideoCapture('/media/macul/hdd/video_sz/group0/output0.avi')
cap = cv2.VideoCapture(args['video_path'])


currentFrame = 0
while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    ret, frame = cap.read()

    if useMtcnn:
        faces = faceDet.getModelOutput(frame)

        if len(faces):
            # face: [ymin, xmin, ymax, xmax]

            face = faces[0] # [ymin, xmin, ymax, xmax]
            p1 = (face[1],face[0])
            p2 = (face[3],face[2])  

            if (p2[0]-p1[0]) < args['min_size'] or (p2[1]-p1[1]) < args['min_size']:
                continue

            if 'COLOR_MOMENT_S3' in config.Feature_Type:
                face_crop, _ = prepare_img(frame, [p1[0], p1[1], p2[0], p2[1]])
            else:
                face_crop = frame[p1[1]:p2[1],p1[0]:p2[0],:].copy()

            #print(face_crop.shape)
            #assert False
            rst = SpoofVal.eval(face_crop, sess, pred, features_tensor)
            rst = np.squeeze(rst[0])
            idx = np.argmax(rst)
            print("spoof: ", idx, rst)
            print('threshold: ', args['threshold'])

            if idx == 0:
                if rst[0]>args['threshold']:
                    frame=cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                elif rst[0]<(1.0-args['threshold_false']):
                    frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
            else:
                if rst[0]<(1.0-args['threshold_false']):
                    frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
            
            #cv2.imshow('frame',face_crop)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            #lbp(gray, p=8, split=3)
            '''
            gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, 8, 2, method="nri_uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            print('hist:', hist.shape)
            plt.hist(lbp.ravel(),bins=range(59))
            plt.show()
            '''
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = faceDet.getModelOutput(frame)
        if len(faces):
            [h, w] = frame.shape[:2]

            face = faces[0] # [ymin, xmin, ymax, xmax]
            p1 = (int(face[1]*w),int(face[0]*h))
            p2 = (int(face[3]*w),int(face[2]*h))  

            if (p2[0]-p1[0]) < args['min_size'] or (p2[1]-p1[1]) < args['min_size']:
                continue

            if 'COLOR_MOMENT_S3' in config.Feature_Type:
                face_crop, _ = prepare_img(frame, [p1[0], p1[1], p2[0], p2[1]])
            else:
                face_crop = frame[p1[1]:p2[1],p1[0]:p2[0],:].copy()

            #print(face_crop.shape)
            #assert False
            rst = SpoofVal.eval(face_crop, sess, pred, features_tensor)
            rst = np.squeeze(rst[0])
            idx = np.argmax(rst)
            print("spoof: ", idx, rst)
            print('threshold: ', args['threshold'])

            if idx == 0:
                if rst[0]>args['threshold']:
                    frame=cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                elif rst[0]<(1.0-args['threshold_false']):
                    frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
            else:
                if rst[0]<(1.0-args['threshold_false']):
                    frame=cv2.rectangle(frame, p1, p2, (255,0,0), 3)
            
            #cv2.imshow('frame',face_crop)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            #lbp(gray, p=8, split=3)
            '''
            gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, 8, 2, method="nri_uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            print('hist:', hist.shape)
            plt.hist(lbp.ravel(),bins=range(59))
            plt.show()
            '''

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
#cap1.release()
cv2.destroyAllWindows()
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

#f_log.close()

# python -m spoof_eval_video -c 7 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96

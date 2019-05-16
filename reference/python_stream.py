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

# spoofing data set
# real face cropped: 
# /black/spoof_db/video_shotbyego/see
# /black/spoof_db/collected/cropped/cropped_scale0.7/positive
# /black/spoof_db/collect_iim/traintest/see

# real face non-cropped: 
# /black/spoof_db/collect_iim/traintest/train_iim
# /black/spoof_db/iim/iim_real_jul_2018
# /black/spoof_db/iim/toukong_real_jul_2018


# video-attack cropped:
# /black/spoof_db/collect_fferr/iim_screrr/traintest/see
# /black/spoof_db/collected/cropped/cropped_scale0.7/screen_attack

# image-attack cropped:
# /black/spoof_db/collected/cropped/cropped_scale0.7/image_attack

# image-attack non-cropped:
# /black/spoof_db/iim/iim_image_attack


# image should be single channel
def lbp(image, p=8, split=1):
    assert len(image.shape)==2, 'only single channel image is supported!'
    assert p in [8,16], 'only 8 and 16 points are supported!'
    if p==8:
        bin_num = 59
    elif p==16:
        bin_num = 243
    if split==1:
        lbp = feature.local_binary_pattern(image, p, 2, method="nri_uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, bin_num))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    else:
        h, w = image.shape
        ph = h/split
        pw = w/split
        hist = np.array([])
        for i in range(split):
            for j in range(split):
                if i == split-1:
                    if j == split-1:
                        lbp = feature.local_binary_pattern(image[i*ph:, j*pw:], p, 2, method="nri_uniform")
                    else:
                        lbp = feature.local_binary_pattern(image[i*ph:, j*pw:(j+1)*pw+2], p, 2, method="nri_uniform")
                else:
                    if j == split-1:
                        lbp = feature.local_binary_pattern(image[i*ph:(i+1)*ph+2, j*pw:], p, 2, method="nri_uniform")
                    else:
                        lbp = feature.local_binary_pattern(image[i*ph:(i+1)*ph+2, j*pw:(j+1)*pw+2], p, 2, method="nri_uniform")
                (hist_tmp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, bin_num))
                hist_tmp = hist_tmp.astype("float")
                hist_tmp /= (hist_tmp.sum() + 1e-7)
                hist = np.append(hist, hist_tmp)
    plt.bar(range(len(hist)),hist)
    plt.show()
    return hist

# https://github.com/jonnedtc/CoALBP/blob/master/methods.py
#mode: + or x
def CoALBP(image, mode='+', lbp_r=1, co_r=2):
    '''
    Co-occurrence of Adjacent Local Binary Patterns algorithm 
    Input image with shape (height, width, channels)
    Input lbp_r is radius for adjacent local binary patterns
    Input co_r is radius for co-occurence of the patterns
    Output features with length 1024 * number of channels
    '''
    h, w, c = image.shape
    # normalize face
    # albp and co-occurrence per channel in image
    histogram = np.empty(0, dtype=np.float)
    for i in range(image.shape[2]):
        C = image[lbp_r:h-lbp_r, lbp_r:w-lbp_r, i].astype(np.float)
        X = np.zeros((4, h-2*lbp_r, w-2*lbp_r)).astype(np.float)
        if mode=='+':
            # adjacent local binary patterns
            X[0, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r      :w-lbp_r,       i] - C
            X[1, :, :] = image[lbp_r      :h-lbp_r,       lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
            X[2, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r      :w-lbp_r,       i] - C
            X[3, :, :] = image[lbp_r      :h-lbp_r,       lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
        else:
            X[0, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
            X[1, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
            X[2, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
            X[3, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
        X = (X>=0).reshape(4, -1)
        # co-occurrence of the patterns
        A = np.dot(np.array([8, 4, 2, 1]), X)
        A = A.reshape(h-2*lbp_r, w-2*lbp_r)
        hh, ww = A.shape
        Y1 = A[0  :hh       , 0  :ww-co_r   ]*16 + A[0      :hh  , co_r :ww     ]
        Y2 = A[0  :hh-co_r  , 0  :ww        ]*16 + A[co_r   :hh  , 0    :ww     ]
        Y3 = A[0  :hh-co_r  , 0  :ww-co_r   ]*16 + A[co_r   :hh  , co_r :ww     ]
        Y4 = A[0  :hh-co_r  , co_r  :ww     ]*16 + A[co_r   :hh  , 0    :ww-co_r]
        Y1 = 1.0*np.bincount(Y1.ravel(), minlength=256)/(hh*(ww-co_r))
        Y2 = 1.0*np.bincount(Y2.ravel(), minlength=256)/(ww*(hh-co_r))
        Y3 = 1.0*np.bincount(Y3.ravel(), minlength=256)/((hh-co_r)*(ww-co_r))
        Y4 = 1.0*np.bincount(Y4.ravel(), minlength=256)/((hh-co_r)*(ww-co_r))        
        pattern = np.concatenate((Y1, Y2, Y3, Y4))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    return histogram


# http://www.ee.oulu.fi/~jkannala/bsif/bsif.html
class BSIF(object):
    def __init__(self, filter_size=11):
        assert filter_size in [3,5,7,9,11,13,15,17], 'filter_size can only be in [3,5,7,9,11,13,1,17]!!'
        self.filter_size = filter_size
        self.filter = loadmat('./texturefilters/ICAtextureFilters_{}x{}_8bit.mat'.format(self.filter_size, self.filter_size))['ICAtextureFilters']
        self.channel = self.filter.shape[2]
        assert self.channel==8, 'only 8-bit filter supported!!'
    def __call__(self,X):
        assert len(X.shape)==2, 'only single channel image supported!!'
        X = X.astype(np.float32)
        y = np.zeros(X.shape)
        for i in range(self.channel):
            tmp = convolve2d(X,self.filter[:,:,i],mode='same',boundary='wrap')
            y += (tmp>0)*(2**i)
        # And finally build the histogram:
        h, b  = np.histogram(y, bins=256, range = (0,256))
        return h / (h.sum() + 1e-7)
    def __repr__(self):
        return "BSIF (filter_size=%s)" % (self.filter_size)

# https://github.com/bytefish/facerec/commit/32aecc9309ddc1267343ddf640303c712a34f775
class LPQ(object):
    """ This implementation of Local Phase Quantization (LPQ) is a 
      Reference: 
        (2008) Blur insensitive texture classification 
        using local phase quantization. Proc. Image and Signal Processing 
    (ICISP 2008), Cherbourg-Octeville, France, 5099:236-243.
        Copyright 2008
    """
    def __init__(self, radius=3):
        self.radius = radius
        self.f = 1.0 # used to control the a in u[0,a] the basic frequency, a = f/n
        self.rho = 0.9 # correlation coefficient
        x = np.arange(-self.radius,self.radius+1)
        n = len(x)
        self.a = self.f/n
        [xp, yp] = np.meshgrid(np.arange(1,(n+1)),np.arange(1,(n+1)))
        pp = np.concatenate((xp,yp)).reshape(2,-1)
        dd = self.euc_dist(pp.T) # squareform(pdist(...)) would do the job, too...
        self.C = np.power(self.rho,dd)
        self.w0 = (x*0.0+1.0)
        self.w1 = np.exp(-2*np.pi*1j*x*self.a)
        self.w2 = np.conj(self.w1)
        self.q1 = self.w0.reshape(-1,1)*self.w1
        self.q2 = self.w1.reshape(-1,1)*self.w0
        self.q3 = self.w1.reshape(-1,1)*self.w1
        self.q4 = self.w1.reshape(-1,1)*self.w2
        u1 = np.real(self.q1)
        u2 = np.imag(self.q1)
        u3 = np.real(self.q2)
        u4 = np.imag(self.q2)
        u5 = np.real(self.q3)
        u6 = np.imag(self.q3)
        u7 = np.real(self.q4)
        u8 = np.imag(self.q4)
        self.M = np.matrix([u1.flatten(1), u2.flatten(1), u3.flatten(1), u4.flatten(1), u5.flatten(1), u6.flatten(1), u7.flatten(1), u8.flatten(1)])
        self.D = np.dot(np.dot(self.M,self.C), self.M.T)
        U,S,V = np.linalg.svd(self.D)
        self.V = V.T
    def euc_dist(self, X):
        Y = X = X.astype(np.float)
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = XX.T
        distances = np.dot(X,Y.T)
        distances *= -2
        distances += XX
        distances += YY
        #np.maximum(distances, 0, distances)
        #distances.flat[::distances.shape[0] + 1] = 0.0
        return np.sqrt(distances)
    def __call__(self,X):
        Qa = convolve2d(X, self.q1, mode='same')
        Qb = convolve2d(X, self.q2, mode='same')
        Qc = convolve2d(X, self.q3, mode='same')
        Qd = convolve2d(X, self.q4, mode='same')
        Fa = np.real(Qa)
        Ga = np.imag(Qa)
        Fb = np.real(Qb) 
        Gb = np.imag(Qb)
        Fc = np.real(Qc) 
        Gc = np.imag(Qc)
        Fd = np.real(Qd) 
        Gd = np.imag(Qd)
        F = np.array([Fa.flatten(1), Ga.flatten(1), Fb.flatten(1), Gb.flatten(1), Fc.flatten(1), Gc.flatten(1), Fd.flatten(1), Gd.flatten(1)])
        G = np.dot(self.V, F)
        t = 0
        # Calculate the LPQ Patterns:
        B = (G[0,:]>=t)*1 + (G[1,:]>=t)*2 + (G[2,:]>=t)*4 + (G[3,:]>=t)*8 + (G[4,:]>=t)*16 + (G[5,:]>=t)*32 + (G[6,:]>=t)*64 + (G[7,:]>=t)*128
        B = np.reshape(B, np.shape(X))
        # And finally build the histogram:
        h, b  = np.histogram(B, bins=256, range = (0,256))
        return h / (h.sum() + 1e-7)
    def __repr__(self):
        return "LPQ (radius=%s)" % (self.radius)


def colorHist(image, location=None, mask=None):
    assert len(image.shape)==2, 'only single channel image is supported!'
    mapping = { 'reye': 0,
                'leye': 1,
                'nose': 2,
                'mouth': 3,
                'r_eyebrow': 4,
                'l_eyebrow': 5,
                'face': 6,
                'btw_eye': 7,
                'btw_eyebrow': 8,
                'nose_tip': 9 }
    if location is None:
        hist = cv2.calcHist([image],[0],None,[256],[0, 256])
        hist = hist.reshape((-1,))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    else:
        assert mask is not None, 'mask cannot be None!'
        hist = np.array([])
        for lc in location:
            m = mask[mapping[lc]]
            print(m)
            #hist_tmp = cv2.calcHist([image[int(m[1]):int(m[3]),int(m[0]):int(m[2])]],[0],None,[256],[0, 256])
            print(image.shape)
            hist_tmp = cv2.calcHist([image[m[1]:m[3],m[0]:m[2]]],[0],None,[256],[0, 256])
            #print(hist_tmp)
            hist_tmp = hist_tmp.astype("float")
            hist_tmp /= (hist_tmp.sum() + 1e-7)
            hist = np.append(hist, hist_tmp)
    #print(hist.shape)
    #plt.bar(range(len(hist)),hist)
    #plt.show()
    return hist


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



def process(color_bgr, depth_array, debug=False):
    #print(color_array.shape)
    #raw_input()
    faces = ssd_det.getModelOutput(color_bgr)
    if len(faces): # only process first face only
        color_bgr = color_bgr.astype(np.uint8)
        landmarks,score = dan_det.detectLandmark(color_bgr, num_points=5, boundingBox=faces[0])        
        p1 = (faces[0][0],faces[0][1])
        p2 = (faces[0][2],faces[0][3])
        if debug:
            print('p1: ', p1)
            print('p2: ', p2)
        #crop_depth_tmp = depth_array[p1[1]:p2[1]+1,p1[0]:p2[0]+1]
        #crop_depth_tmp = crop_depth_tmp*depth_scale 
        STM = similar_transform_matrix(landmarks, cfg['sim_tr_dst_pts'])
        crop_color = cv2.warpAffine(color_bgr, STM, cfg['tgt_face_size'])
        crop_depth = cv2.warpAffine(depth_array, STM, cfg['tgt_face_size'])
        if debug:
            print('max crop_depth: ', np.max(crop_depth))
            print('min crop_depth: ', np.min(crop_depth))
            print('median crop_depth: ', np.median(crop_depth))

        points, points_5, area = landmark_area(dan_det.detectLandmark, crop_color)
        clean_up_depth_data1(crop_depth, points[0:17,:])
        new_origin = find_nose_peak_xy(crop_depth, area) # get the center point index, start from 0
        test_face = align_faces_3d_new1(crop_depth,new_origin)
        test_face_roll = face_y_roll(test_face,points)
        return test_face_roll, test_face, points, points_5, area, crop_color, crop_depth, faces[0], new_origin
    else:
        return None, None, None, None, None, None, None, None, None

cfg = {
    'tgt_face_size': tuple(np.array([96, 96])*1-12),
    'sim_tr_dst_pts': np.array([[30.29,51.69-20],[65.53,51.5-20],[48.02,71.73-20],[33.55,92.37-20],[62.73,92.2-20]])*1-6,
    'ssd_model_path': '/home/macul/Projects/ego_mk_op/ego/detection_ssd/models/',
}


ssd_det = ssd(cfg['ssd_model_path'], mode='gpu1', img_height = 480, img_width = 640)
dan_det = dan(mode='gpu1')


#f_log = open('/home/macul/Projects/realsense/distance_face_size.log','wb')

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

currentFrame = 0
while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    ret, frame = cap.read()    
    ret1, frame1 = cap1.read()

    faces = ssd_det.getModelOutput(frame)
    if len(faces):
        # area: [x1,y1,x2,y2] for reye, leye, nose, mouth, r_eyebrow, l_eyebrow, face, btw eyes, btw eyebrows, nose tip
        points, points_5, area = landmark_area(dan_det.detectLandmark, frame, faces[0])
        colorHist(frame[:,:,0],['reye','leye'],area)
        #landmarks,score = dan_det.detectLandmark(frame, num_points=5, boundingBox=faces[0])
        STM = similar_transform_matrix(points_5, cfg['sim_tr_dst_pts'])
        crop_color = cv2.warpAffine(frame, STM, cfg['tgt_face_size'])
        p1 = (faces[0][0],faces[0][1])
        p2 = (faces[0][2],faces[0][3])
        frame=cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        dan_det.drawPoints(frame, points)
        #print('frame shape: ', frame[:,:,0].shape)
        gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
        
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

    faces1 = ssd_det.getModelOutput(frame1)
    if len(faces1):
        landmarks1,score1 = dan_det.detectLandmark(frame1, num_points=5, boundingBox=faces1[0])
        STM1 = similar_transform_matrix(landmarks1, cfg['sim_tr_dst_pts'])
        crop_color1 = cv2.warpAffine(frame1, STM1, cfg['tgt_face_size'])
        p11 = (faces1[0][0],faces1[0][1])
        p22 = (faces1[0][2],faces1[0][3])
        frame1=cv2.rectangle(frame1, p11, p22, (0,255,0), 3)

    if False and len(faces) and len(faces1):
        combine_crop = np.concatenate((crop_color, crop_color1),axis=1)
        cv2.imshow('frame',combine_crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        combine = np.concatenate((frame,frame1),axis=1)

        cv2.imshow('frame',combine)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    currentFrame += 1

# When everything done, release the capture
cap.release()
cap1.release()
cv2.destroyAllWindows()
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

#f_log.close()
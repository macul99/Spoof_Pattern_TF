from PIL import Image
import numpy as np
import cv2
from skimage import feature
from scipy.signal import convolve2d
from scipy.io import loadmat
from scipy.stats import skew

# image should be single channel
def lbp(image, p=8, r=1, split=1):
    assert len(image.shape)==2, 'only single channel image is supported!'
    assert p in [8,16], 'only 8 and 16 points are supported!'
    if p==8:
        bin_num = 59
    elif p==16:
        bin_num = 243
    if split==1:
        lbp = feature.local_binary_pattern(image, p, r, method="nri_uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, bin_num))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    else:
        h, w = image.shape
        ph = int(h/split)
        pw = int(w/split)
        hist = np.array([])
        for i in range(split):
            for j in range(split):
                if i == split-1:
                    if j == split-1:
                        lbp = feature.local_binary_pattern(image[i*ph:, j*pw:], p, r, method="nri_uniform")
                    else:
                        lbp = feature.local_binary_pattern(image[i*ph:, j*pw:(j+1)*pw+2], p, r, method="nri_uniform")
                else:
                    if j == split-1:
                        lbp = feature.local_binary_pattern(image[i*ph:(i+1)*ph+2, j*pw:], p, r, method="nri_uniform")
                    else:
                        lbp = feature.local_binary_pattern(image[i*ph:(i+1)*ph+2, j*pw:(j+1)*pw+2], p, r, method="nri_uniform")
                (hist_tmp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, bin_num))
                hist_tmp = hist_tmp.astype("float")
                hist_tmp /= (hist_tmp.sum() + 1e-7)
                hist = np.append(hist, hist_tmp)
    #plt.bar(range(len(hist)),hist)
    #plt.show()
    return hist

# LBP applied to grey scale image
class LBP_GREY(object):
    def __init__(self):
        return
    def __call__(self,X):
        if len(X.shape) == 3:
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        hist = lbp(X, p=8, r=1, split=3)
        hist = np.append(hist, lbp(X, p=8, r=2, split=1))
        hist = np.append(hist, lbp(X, p=16, r=2, split=1))
        return hist

# Red channel LBP minus Green channel LBP
class LBP_RG(object):
    def __init__(self):
        return
    def __call__(self,X):
        assert len(X.shape) == 3, "BGR image is expected!!!"
        hist_r = lbp(X[:,:,2], p=8, r=1, split=3)
        hist_g = lbp(X[:,:,1], p=8, r=1, split=3)
        return np.abs(hist_r-hist_g)

# https://github.com/jonnedtc/CoALBP/blob/master/methods.py
#mode: + or x
def coAlbp(image, mode='+', lbp_r=1, co_r=2):
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

class CoALBP(object):
    def __init__(self):
        return
    def __call__(self,X):
        assert len(X.shape) == 3, "color image is expected!!!"
        hist = coAlbp(X)
        hist = np.append(hist, coAlbp(X, mode='x'))
        return hist

# vector length is 256 for each mode
class CoALBP_256(object):
    def __init__(self):
        return
    def __call__(self,X):
        assert len(X.shape) == 3, "color image is expected!!!"
        temp = coAlbp(X)
        hist = np.sum(coAlbp(X).reshape((-1,256)), axis=0)
        hist = np.append(hist, np.sum(coAlbp(X, mode='x').reshape((-1,256)), axis=0))
        return hist

class CoALBP_GREY(object):
    def __init__(self):
        return
    def __call__(self,X):
        assert len(X.shape) == 3, "color image is expected!!!"
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        X = X[:,:,np.newaxis]
        hist = coAlbp(X)
        hist = np.append(hist, coAlbp(X, mode='x'))
        return hist

# http://www.ee.oulu.fi/~jkannala/bsif/bsif.html
class BSIF(object):
    def __init__(self, filter_size=11):
        assert filter_size in [3,5,7,9,11,13,15,17], 'filter_size can only be in [3,5,7,9,11,13,1,17]!!'
        self.filter_size = filter_size
        self.filter = loadmat('./texturefilters/ICAtextureFilters_{}x{}_8bit.mat'.format(self.filter_size, self.filter_size))['ICAtextureFilters']
        self.channel = self.filter.shape[2]
        assert self.channel==8, 'only 8-bit filter supported!!'
    def __call__(self,X):
        if len(X.shape) == 3:
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
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
        if len(X.shape) == 3:
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
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

# http://www.ee.oulu.fi/~jkannala/bsif/bsif.html
class COLOR_MOMENT(object):
    def __init__(self, split=2):
        self.split = split
    def get_color_moment(self, X):
        h, w, c = X.shape
        moment = []
        for i in range(c):
            hist = colorHist(X[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]
        return moment
    def __call__(self,X):
        assert len(X.shape) == 3, "BGR image is expected!!!"
        X = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)        
        if self.split==1:
            moment = self.get_color_moment(X)
        else:
            h, w, c = X.shape
            ph = int(h/self.split)
            pw = int(w/self.split)
            moment = []
            for i in range(self.split):
                for j in range(self.split):
                    if i == self.split-1:
                        if j == self.split-1:
                            moment += self.get_color_moment(X[i*ph:, j*pw:, :])
                        else:
                            moment += self.get_color_moment(X[i*ph:, j*pw:(j+1)*pw, :])
                    else:
                        if j == self.split-1:
                            moment += self.get_color_moment(X[i*ph:(i+1)*ph, j*pw:, :])
                        else:
                            moment += self.get_color_moment(X[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :])
        return np.array(moment)
    def __repr__(self):
        return "COLOR_MOMENT (split=%s)" % (self.split)


# http://www.ee.oulu.fi/~jkannala/bsif/bsif.html
# multiple color space without split
class COLOR_MOMENT2(object):
    def __init__(self):
        return
    def __call__(self,X):
        assert len(X.shape) == 3, "BGR image is expected!!!"
        h, w, c = X.shape
        moment = []

        # BGR
        for i in range(c):
            hist = colorHist(X[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]

        # HSV, V is intensity
        HSV = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
        for i in [0,1]:
            hist = colorHist(HSV[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]

        # YCrCb, Y is intensity
        YCrCb = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
        for i in [1,2]:
            hist = colorHist(YCrCb[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]

        # LAB, L is intensity
        LAB = cv2.cvtColor(X, cv2.COLOR_BGR2LAB)
        for i in [1,2]:
            hist = colorHist(LAB[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]
        return np.array(moment)


# http://www.ee.oulu.fi/~jkannala/bsif/bsif.html
# HSV+YCrCb+LAB with split
class COLOR_MOMENT3(object):
    def __init__(self, split=2):
        self.split = split
    def get_color_moment(self, X):
        h, w, c = X.shape
        moment = []
        X1 = cv2.cvtColor(X, cv2.COLOR_BGR2HSV) 
        for i in [0,1]:
            hist = colorHist(X1[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]
        X2 = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb) 
        for i in [1,2]:
            hist = colorHist(X2[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]
        X3 = cv2.cvtColor(X, cv2.COLOR_BGR2LAB) 
        for i in [1,2]:
            hist = colorHist(X3[:,:,i])
            moment += [np.mean(hist), np.std(hist), skew(hist)]
        return moment
    def __call__(self,X):
        assert len(X.shape) == 3, "BGR image is expected!!!"
        if self.split==1:
            moment = self.get_color_moment(X)
        else:
            h, w, c = X.shape
            ph = int(h/self.split)
            pw = int(w/self.split)
            moment = []
            for i in range(self.split):
                for j in range(self.split):
                    if i == self.split-1:
                        if j == self.split-1:
                            moment += self.get_color_moment(X[i*ph:, j*pw:, :])
                        else:
                            moment += self.get_color_moment(X[i*ph:, j*pw:(j+1)*pw, :])
                    else:
                        if j == self.split-1:
                            moment += self.get_color_moment(X[i*ph:(i+1)*ph, j*pw:, :])
                        else:
                            moment += self.get_color_moment(X[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :])
        return np.array(moment)
    def __repr__(self):
        return "COLOR_MOMENT (split=%s)" % (self.split)
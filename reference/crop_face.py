from imutils import paths
import numpy as np
import cv2
import os
from ssd import ssd
from os import listdir, makedirs
from os.path import join, exists

folder_dic = {	#'/media/macul/black/spoof_db/collect_iim/traintest/train_iim': False,
				'/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/test': False,
				'/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/train': False,
				#'/media/macul/black/spoof_db/iim/iim_real_jul_2018': True,
				#'/media/macul/black/spoof_db/iim/toukong_real_jul_2018': True,
				#'/media/macul/black/spoof_db/iim/iim_image_attack': True,
			 }

ssd_det = ssd('/home/macul/Projects/ego_mk_op/ego/detection_ssd/models/', mode='gpu1', resize_ratio = 1.0)

def scale_face(bbox, scale = 0.8):
	assert scale < 1, 'scale must be less than 1!!!'
	x1 = 1.0*bbox[0]
	y1 = 1.0*bbox[1]
	x2 = 1.0*bbox[2]
	y2 = 1.0*bbox[3]
	dx = round((x2-x1+1)*scale/2)
	dy = round((y2-y1+1)*scale/2)
	xc = round(np.mean([x1,x2]))
	yc = round(np.mean([y1,y2]))
	return [int(round(xc-dx)), int(round(yc-dy)), int(round(xc+dx)), int(round(yc+dy))]

for k, v in folder_dic.iteritems():
	dir_name = k+'_cropped'
	if not exists(dir_name):
		makedirs(dir_name)	
	if v:
		face_scale = 0.99

		with open(k+'_processed/bbox.txt', 'rb') as f:
			f.readline()
			for line in f.readlines():
				fn, x1, y1, w, h = line.split()
				bbox = [int(x1), int(y1), int(x1)+int(w)-1, int(y1)+int(h)-1]
				sbbox = scale_face(bbox, face_scale)

				img = cv2.imread(join(k, fn))
				print('img shape: ', img.shape)
				cv2.imwrite(join(dir_name, fn), img[sbbox[1]:sbbox[3],sbbox[0]:sbbox[2],:])
	else:
		face_scale = 0.99
		imagePaths = list(paths.list_images(k))
		for fn in imagePaths:
			img = cv2.imread(fn)
			faces = ssd_det.getModelOutput(img)
			if len(faces):
				sbbox = scale_face(faces[0], face_scale)
				cv2.imwrite(join(dir_name, fn.split('/')[-1]), img[sbbox[1]:sbbox[3],sbbox[0]:sbbox[2],:])
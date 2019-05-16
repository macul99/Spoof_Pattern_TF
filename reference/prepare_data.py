# obsolete
import numpy as np
from scipy.optimize import minimize
import time
import os
from os import listdir, makedirs
from os.path import join, exists
import pickle
import pandas as pd

folder_dic = {  '/media/macul/black/spoof_db/collect_iim/traintest/train_iim_cropped'			: {'class': 0, 'dataset': 'real_1'}, # 7641
                '/media/macul/black/spoof_db/collect_iim/traintest/see'							: {'class': 0, 'dataset': 'real_2'}, # 795
                '/media/macul/black/spoof_db/iim/iim_real_jul_2018_cropped'						: {'class': 0, 'dataset': 'real_3'}, # 12386
                '/media/macul/black/spoof_db/iim/toukong_real_jul_2018_cropped'					: {'class': 0, 'dataset': 'real_4'}, # 6783
                '/media/macul/black/spoof_db/video_shotbyego/see'								: {'class': 0, 'dataset': 'real_5'}, # 4107
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/positive/iim'	: {'class': 0, 'dataset': 'real_6'}, # 59
                '/media/macul/black/spoof_db/iim/iim_image_attack_cropped'						: {'class': 1, 'dataset': 'image_attack_1'}, # 271
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/image_attack'	: {'class': 1, 'dataset': 'image_attack_2'}, # 445
                '/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/test_cropped'	: {'class': 1, 'dataset': 'image_attack_3'}, # 2142
                '/media/macul/black/spoof_db/collect_fferr/iim_imgerr/traintest/train_cropped'	: {'class': 1, 'dataset': 'image_attack_4'}, # 11282
                '/media/macul/black/spoof_db/collected/cropped/cropped_scale0.7/screen_attack'	: {'class': 2, 'dataset': 'video_attack_1'}, # 1069
                '/media/macul/black/spoof_db/collect_fferr/iim_screrr/traintest/see'			: {'class': 2, 'dataset': 'video_attack_2'}, # 10124
             }


data_dic = {'data_source': [	'/media/macul/black/spoof_db/feature_extraction/real_1.pkl',
								'/media/macul/black/spoof_db/feature_extraction/real_3.pkl',
								'/media/macul/black/spoof_db/feature_extraction/image_attack_2.pkl',
								'/media/macul/black/spoof_db/feature_extraction/image_attack_4.pkl',
								'/media/macul/black/spoof_db/feature_extraction/video_attack_2.pkl',
								],
			'feature_list': ['LPQ','BSIF','LBP_GREY','CoALBP','LBP_RG','COLOR_MOMENT'],
			}

data_dic = {'data_source': [	'/media/macul/black/spoof_db/feature_extraction/image_attack_3.pkl',
								],
			'feature_list': ['LPQ','BSIF','LBP_GREY','CoALBP','LBP_RG','COLOR_MOMENT'],
			}

for file in data_dic['data_source']:
	data = []
	labels = []
	with open(file,'rb') as f:
	    input_dic = pickle.load(f)
	    input_ft = input_dic['feature']
	    class_label = input_dic['class']
	    for index, row in input_ft.iterrows():
	    	print(index)	    	
	    	ft = []
	    	for d in data_dic['feature_list']:
	    		ft += list(row[d])
	    	if np.NaN not in ft:
	    		data += [ft]
	    		labels += [class_label]

data_dic['data'] = data
data_dic['labels'] = labels

with open('/media/macul/black/spoof_db/feature_extraction/training_1.pkl','wb') as f:
	pickle.dump(data_dic,f,pickle.HIGHEST_PROTOCOL)
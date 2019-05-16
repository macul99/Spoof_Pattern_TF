import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
import tensorflow as tf
import numpy as np
import argparse
import logging
import json
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from tensorflow.core.protobuf import config_pb2
from PIL import Image
import pickle
#import sys
#sys.path.append('/home/macul/libraries/mk_utils/mklib/nn/tfnet/')
#sys.path.append('/home/macul/libraries/mk_utils/mklib/nn/tfloss/')
#sys.path.append('/home/macul/libraries/mk_utils/tf_spoof/')
#sys.path.append('/home/macul/libraries/mk_utils/spoofing_lbp/')
from spoof_feature import LBP_GREY, LBP_RG, CoALBP, CoALBP_256, CoALBP_GREY, BSIF, LPQ, COLOR_MOMENT, COLOR_MOMENT2, COLOR_MOMENT3
#from tf_spoof.config import net_config as config
from locallib.nn.tfnet.tfspoofdense import SpoofDenseNet
from locallib.nn.tfloss.tfloss import TfLosses


class SpoofDspTf():
    def __init__(self, config):
        self.config = config

        self.Model_Process = {  'BSIF'              : BSIF(),
                                'CoALBP'            : CoALBP(),
                                'CoALBP_GREY'       : CoALBP_GREY(),
                                'CoALBP_256'        : CoALBP_256(),
                                'COLOR_MOMENT'      : COLOR_MOMENT(),
                                'COLOR_MOMENT2'     : COLOR_MOMENT2(),
                                'COLOR_MOMENT3'     : COLOR_MOMENT3(),
                                'COLOR_MOMENT_S3'   : COLOR_MOMENT(split=3),
                                'LBP_GREY'          : LBP_GREY(),
                                'LBP_RG'            : LBP_RG(),
                                'LPQ'               : LPQ(),
                             }

        # get ft extractor and network configuration
        self.ft_extractor = [self.Model_Process[ex] for ex in self.config.Feature_Type]
        self.ft_layer_units = [self.config.Dense_Cfg[ex] for ex in self.config.Feature_Type]

        # declare placeholders for training
        self.labels_ph = tf.placeholder(name='label',shape=[None,], dtype=tf.int64)
        self.features_ph = tuple([tf.placeholder(name='ft_{}'.format(f_key),shape=[None,self.config.Feature_Size[f_key]], 
                                                 dtype=tf.float32) for f_key in self.config.Feature_Type])
        # declare placeholders for validation
        self.labels_val_ph = tf.placeholder(name='label_val',shape=[None,], dtype=tf.int64)
        self.features_val_ph = tuple([tf.placeholder(name='ft_val_{}'.format(f_key),shape=[None,self.config.Feature_Size[f_key]], 
                                                     dtype=tf.float32) for f_key in self.config.Feature_Type])
        # declare placeholders for evaluation
        self.img_ph = tf.placeholder(name='image_val',shape=[None,None,3], dtype=tf.uint8)
        # create output folder
        self.checkpointsPath = os.path.sep.join([self.config.OUT_PATH, self.config.PREFIX])
        if not isdir(self.checkpointsPath):
            mkdir(self.checkpointsPath)

    def build_dataset(self, rec_path, batch_size, classes, training=True):
        if training:
            img_shuffle_grid_size = self.config.Img_shuffle_grid_size
        else:
            img_shuffle_grid_size = 0

        dataset = tf.data.TFRecordDataset(rec_path)
        dataset = dataset.map(lambda x: self.parse_function(x, img_shuffle_grid_size, classes))
        if training:
            dataset = dataset.shuffle(buffer_size=self.config.BUFFER_SIZE) # shuffle the whole dataset is better
        dataset = dataset.map(lambda *x: (self.feature_extraction(x[0], self.ft_extractor), x[1]))
        # remember to set batch size to 1 for dataset debug
        #dataset = dataset.map(lambda *x: self.feature_extraction(x[0], self.ft_extractor) + (x[1],)) # for dataset debug
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element

    def build_net(self, features_ph, labels_ph, training=True, reuse=None):
        if training:
            arc_margin = self.config.Arc_margin_angle
        else:
            arc_margin = 0

        with tf.variable_scope(self.config.NET_SCOPE, reuse=reuse):
            # build base network
            embeddings = SpoofDenseNet.build(features_ph, feature_units=self.ft_layer_units, stem_units=self.config.Dense_Stem_Cfg, 
                                             training=training, act=self.config.Activation, reg=self.config.Regularizer, 
                                             init=self.config.Initializer, scope='spoof')            
            if self.config.LOSS_TYPE=='arc':
                logit, inference_loss = TfLosses.arc_loss(embedding=embeddings, labels=labels_ph, w_init=self.config.Initializer, 
                                                          out_num=self.config.NUM_CLASSES, s=self.config.Arc_margin_scale, 
                                                          m=arc_margin)
            else:
                logit, inference_loss = TfLosses.softmax_loss(embedding=embeddings, labels=labels_ph, out_num=self.config.NUM_CLASSES,  
                                                              act=self.config.Activation, reg=self.config.Regularizer, 
                                                              init=self.config.Initializer)
            pred = tf.nn.softmax(logit, name='prediction') # output name: 'SpoofDenseNet/prediction'
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels_ph), dtype=tf.float32))
        return embeddings, logit, inference_loss, pred, acc

    def train_net(self, start_epoch=-1):
        # build training dataset and network
        iterator_train, next_element_train = self.build_dataset(self.config.TRAIN_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, training=True)
        embeddings, logit, inference_loss, pred, acc = self.build_net(self.features_ph, self.labels_ph, training=True, reuse=None)

        # prepare for training
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        lr = tf.train.piecewise_constant(global_step, boundaries=self.config.Opt_lr_steps, values=self.config.Opt_lr, name='lr_schedule')
        # define the optimize method
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.config.Opt_momentum)
        # get train op
        #grads = opt.compute_gradients(inference_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):        
            #train_op = opt.apply_gradients(grads, global_step=global_step)
            train_op = opt.minimize(inference_loss, global_step=global_step)

        # build validation dataset and network
        iterator_val, next_element_val = self.build_dataset(self.config.VAL_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, training=False)
        embeddings_val, logit_val, _, pred_val, acc_val = self.build_net(self.features_val_ph, self.labels_val_ph, training=False, reuse=True)


        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        # create model saver
        saver = tf.train.Saver(max_to_keep=50)        

        #time_stamp = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))           

        # define log file
        log_file_path = os.path.join(self.checkpointsPath, 'train_{}'.format(start_epoch+1) + '.log')
        log_file = open(log_file_path, 'w')        

        # summary writer
        summary = tf.summary.FileWriter(self.checkpointsPath, sess.graph)
        summaries = []
        # trainabel variable gradients
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # add loss summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        # add learning rate
        summaries.append(tf.summary.scalar('leraning_rate', lr))
        # add accuracy
        summaries.append(tf.summary.scalar('accuracy', acc))
        summary_op = tf.summary.merge(summaries)


        # start training process
        img_verify_flag = False # set to false during actual training
        count = 0
        validation_result = []
        
        if start_epoch >= 0:
            # restore checkpoint
            saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(start_epoch))
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(self.config.NUM_EPOCH):
            
            sess.run(iterator_train.initializer)
            while True:
                try:
                    start = time.time()
                    images_train, labels_train = sess.run(next_element_train)
                    #images_train, images_show, labels_train = sess.run(next_element_train) # for dataset debug
                    if img_verify_flag: # for dataset debug
                        img = Image.fromarray(images_show[0,...], 'RGB')
                        img.show()
                        exit(1)
                    feed_dict = {self.features_ph: images_train, self.labels_ph: labels_train}            
                    embTrain, logitTrain, inferenceLossTrain, accTrain, _, _ = \
                        sess.run([embeddings, logit, inference_loss, acc, train_op, inc_op],
                                  feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                    end = time.time()
                    pre_sec = self.config.BATCH_SIZE/(end - start)
                    
                    if count > 0 and count % self.config.Update_Interval == 0:
                        # logging
                        print('Training: epoch %d, total_step %d, inference loss is %.2f, '
                              'accuracy is %.6f, time %.3f samples/sec' %
                                  (i, count, inferenceLossTrain, accTrain,pre_sec))
                        log_file.write('Training: epoch %d, total_step %d, inference_loss %.2f, '
                                       'accuracy %.6f, time %.3f samples/sec' %
                                       (i, count, inferenceLossTrain, accTrain,pre_sec) + '\n')
                        log_file.flush()
                        
                        # save summary
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)
                        
                        #print(embTrain)
                        #print(logitTrain)
                        #print(inferenceLossTrain)
                        #print(labels_train)
                        #print(images_train)
                    count += 1
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
            
            # save check points
            ckpt_filename = self.config.PREFIX+'_{:d}'.format(i) + '.ckpt'
            ckpt_filename = os.path.join(self.checkpointsPath, ckpt_filename)
            saver.save(sess, ckpt_filename)
            print('######### Save checkpoint: {}'.format(ckpt_filename))
            log_file.write('Checkpoint: {}'.format(ckpt_filename) + '\n')
            log_file.flush()
            
                    
            # do validation
            accVal = []
            predVal = []
            labelVal = np.array([])
            sess.run(iterator_val.initializer)
            while True:
                try:
                    images_val, labels_val = sess.run(next_element_val)
                    feed_dict = {self.features_val_ph: images_val, self.labels_val_ph: labels_val}
                    acc_tmp, pred_tmp = \
                        sess.run([acc_val, pred_val],
                                  feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))                        
                    accVal += [acc_tmp]
                    #print(accVal)
                    
                    if type(predVal) == type([]):
                        predVal = pred_tmp
                    else:
                        predVal = np.append(predVal, pred_tmp, axis=0)
                    labelVal = np.append(labelVal, labels_val)
                    
                    #print(predVal)
                    #print(labelVal)
                except tf.errors.OutOfRangeError:
                    break    
            accVal = np.mean(accVal)
            print('$$$$$$$$ Validation: epoch %d, accuracy is %.6f' % (i, accVal))
            log_file.write('Validation: epoch %d, accuracy %.6f' % (i, accVal) + '\n')
            log_file.flush()
            
            # save validation results
            validation_result += [{'label': labelVal, 'pred': predVal}]
            with open(self.checkpointsPath+'/'+self.config.PREFIX+'_val_result.pkl', 'wb') as f:
                pickle.dump(validation_result, f)
        log_file.close()

    def validate_net(self, checkpoint_num, output_pickle_path):
        out_dir = output_pickle_path[0:output_pickle_path.rfind('/')]
        if not isdir(out_dir):
            mkdir(out_dir)

        # build validation dataset and network
        iterator_val, next_element_val = self.build_dataset(self.config.VAL_REC, self.config.BATCH_SIZE, self.config.NUM_CLASSES, training=False)
        embeddings_val, logit_val, _, pred_val, acc_val = self.build_net(self.features_val_ph, self.labels_val_ph, training=False, reuse=None)


        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        # create model saver
        saver = tf.train.Saver()
        
        # restore checkpoint
        saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

        # do validation
        accVal = []
        predVal = []
        labelVal = np.array([])
        sess.run(iterator_val.initializer)

        while True:
            try:
                images_val, labels_val = sess.run(next_element_val)
                feed_dict = {self.features_val_ph: images_val, self.labels_val_ph: labels_val}
                acc_tmp, pred_tmp = \
                    sess.run([acc_val, pred_val],
                              feed_dict=feed_dict,
                              options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))                        
                accVal += [acc_tmp]
                #print(accVal)
                
                if type(predVal) == type([]):
                    predVal = pred_tmp
                else:
                    predVal = np.append(predVal, pred_tmp, axis=0)
                labelVal = np.append(labelVal, labels_val)
                
                #print(predVal)
                #print(labelVal)
            except tf.errors.OutOfRangeError:
                break    
        accVal = np.mean(accVal)
        print('$$$$$$$$ Validation accuracy is %.6f' % (accVal))
        
        # save validation results
        validation_result = {'label': labelVal, 'pred': predVal}
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(validation_result, f)

        return validation_result

    def deploy_net(self, checkpoint_num):
        # build deploy network
        embeddings, logit, _, pred, acc = self.build_net(self.features_val_ph, self.labels_val_ph, training=False, reuse=None)

        features_tensor = self.feature_extraction(self.img_ph, self.ft_extractor)

        # create model saver
        saver = tf.train.Saver()

        # create session
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)#, device_count={'CPU' : 1, 'GPU' : 0})
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        
        # restore checkpoint
        saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

        return sess, embeddings, logit, pred, acc, features_tensor

    # img should be in RGB order
    def eval(self, img, sess, pred, features_tensor, embeddings=None, logit=None):
        #features_tensor = self.feature_extraction(img, self.ft_extractor)
        features = sess.run(features_tensor, feed_dict={self.img_ph: img}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
        #features = tuple([np.ones(36)])
        features_new = []

        for i, ft in enumerate(features):
            features_new += [ft[np.newaxis,:]]

        features_new = tuple(features_new)

        feed_dict = {self.features_val_ph: features_new, self.labels_val_ph: [0]}

        pred_eval = sess.run([pred], feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)) 

        return pred_eval

    def img_crop_shuffle(self, img, height, width, grid_size=4):
        crop_h = tf.cast(height/grid_size,tf.int32)
        crop_w = tf.cast(width/grid_size,tf.int32)

        idx = list(range(grid_size*grid_size))
        np.random.shuffle(idx)
        img_tmp = []
        cnt = 0
        for i, v in enumerate(idx):
            if i%grid_size == 0:
                img_crop = tf.image.crop_to_bounding_box(img, int(idx[i]/grid_size)*crop_h,int(idx[i]%grid_size)*crop_w,crop_h,crop_w)
            else:
                img_crop = tf.concat([img_crop, tf.image.crop_to_bounding_box(img, int(idx[i]/grid_size)*crop_h,int(idx[i]%grid_size)*crop_w,crop_h,crop_w)], 1)
            cnt += 1
            if cnt == grid_size:
                img_tmp += [img_crop]
                cnt = 0
        img1 = img_tmp[0]
        for i in range(1,len(img_tmp)):
            img1 = tf.concat([img1,img_tmp[i]],0)
            
        return img1

    # this function must be used for batch_size of 1 or before batch operation since the image size varies
    def parse_function(self, example_proto, grid_size=4, classes=3):
        assert classes in [2,3], 'only classes of 2 or 3 supported!!!'
        
        features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                    'image/height': tf.FixedLenFeature([], tf.int64),
                    'image/width': tf.FixedLenFeature([], tf.int64),
                    #'image/colorspace': tf.FixedLenFeature([], tf.string),
                    #'image/channels': tf.FixedLenFeature([], tf.int64),
                    #'image/class/text': tf.FixedLenFeature([], tf.string),
                    'image/class/label': tf.FixedLenFeature([], tf.int64)}
        
        features = tf.parse_single_example(example_proto, features)
        
        img = tf.image.decode_jpeg(features['image/encoded'])
        #img = features['image/encoded']
        
        # img = tf.reshape(img, shape=(112, 112, 3))
        # r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
        # img = tf.concat([b, g, r], axis=-1)
        
        #img = tf.cast(img, dtype=tf.float32)
        #img = tf.subtract(img, 127.5)
        #img = tf.multiply(img,  0.0078125)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        
        if classes==3:
            label = tf.cast(features['image/class/label'], tf.int64) - 1
        else:
            label = tf.cast(features['image/class/label']>1, tf.int64) # two class only
        
        if grid_size <= 1:
            return img, label
        
        height = tf.cast(features['image/height'], tf.int64)
        width = tf.cast(features['image/width'], tf.int64)
        
        return self.img_crop_shuffle(img,height,width,grid_size), label

    def feature_extraction(self, img, extractor):
        ft = []
        for ex in extractor:
            ft += [tf.cast(tf.py_func(ex, [img], tf.double),tf.float32)]
        return tuple(ft) # list does not work here, need to convert to tuple
        #return tuple(ft), img # for dataset debug

    def get_output_name(self):
        sess, embeddings, logit, pred, acc, features_tensor = self.deploy_net(0)
        return [n.name for n in tf.get_default_graph().as_graph_def().node]

    def save_pb(self, checkpoint_num, output_name=['SpoofDenseNet/prediction']):
        with tf.Session() as sess:
            # restore graph
            saver = tf.train.import_meta_graph(self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt.meta'.format(checkpoint_num))
            # load weight
            saver.restore(sess, self.checkpointsPath+'/'+self.config.PREFIX+'_{}.ckpt'.format(checkpoint_num))

            # Freeze the graph
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_name)

            # Save the frozen graph
            with open(self.checkpointsPath+'/'+self.config.PREFIX+'_{}.pb'.format(checkpoint_num), 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())

        

'''
# to get output name
import tensorflow as tf
import numpy as np
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
from spoofing_lbp.SpoofDspTf import SpoofDspTf
from tf_spoof.config import net_config as config
SpoofVal = SpoofDspTf(config)
sess, embeddings, logit, pred, acc, features_tensor = SpoofVal.deploy_net(0)
output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
'''
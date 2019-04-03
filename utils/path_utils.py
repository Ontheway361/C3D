#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/30
author: lujie
"""

import os
import time
from IPython import embed

# root_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')))

# root_dir = '/home/gpu3/lujie/video_understanding/dataset'
root_dir = '/home/lujie/Documents/deep_learning/video_understanding/'

class PathSet(object):

    @staticmethod
    def root_dir():
        return root_dir

    @staticmethod
    def log_dir():
        ''' get the log dir '''

        stamp  = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        logdir = os.path.join(root_dir + 'log/log_dir/' + stamp)
        return logdir


    @staticmethod
    def model_dir(model_name = 'C3D', cur_epochs = 0):
        ''' get the root path of video_understanding '''

        save_dir = os.path.join(root_dir, 'saved_model/saved_c3d_models', model_name + '_epoch-' + str(cur_epochs - 1) + '.pth.tar')
        return save_dir


    @staticmethod
    def db_dir(database = 'ucf101'):
        ''' get the original and processed target folder dir '''

        dataset_dir, output_dir = None, None

        if database == 'ucf101':
            dataset_dir = root_dir + 'dataset/ucf101_related/UCF-101'
            output_dir  = root_dir + 'dataset/ucf101_related/processed_ucf101'
            return dataset_dir, output_dir
        elif database == 'hmdb51':
            dataset_dir = root_dir + 'dataset/hmdb51_related/hmdb-51'
            output_dir  = root_dir + 'dataset/hmdb51_related/processed_hmdb51'
            return dataset_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError


    @staticmethod
    def pretrained_model_dir():
        ''' get the dir of pretrained model '''

        pretrained_dir = root_dir + '/saved_model/pretrained_model/c3d_pretrained.pth'
        return pretrained_dir


    @staticmethod
    def label_dir():
        ''' Get the label_path '''

        label_path = root_dir + 'dataset/ucf101_related/ucf_labels.txt'
        return label_path

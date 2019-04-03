#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/03/30
author: lujie
"""


import torch 
from runner.c3d_runner import C3D_Train

if __name__ == '__main__':
    
    params_dict = {
   
        'dataset'     : 'ucf101',
        'num_classes' : 101,   
        'model_name'  : 'C3D',
        'pretrained'  : True, 
        'num_epochs'  : 20,
        'resume_epoch': 15,   # default : 15
        'frame_mode'  : 1,  # 0 : continuous  |  1 : uniform intervals 
        'clip_len'    : 16,
        'batch_size'  : 8,
        'save_freq'   : 5,
        'lr'          : 3.5e-4,    # TODO
        'useTest'     : True,
        'device'      : torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    }
   
    model_engine = C3D_Train(params_dict)
   
    model_engine.model_train()
   
   

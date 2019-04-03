#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/01
author: lujie
"""

import torch 


params_dict = {
   
    'dataset'     : 'ucf101',
    'num_classes' : 101,   
    'model_name'  : 'C3D',
    'pretrained'  : True, 
    'num_epochs'  : 100,
    'resume_epoch': 0,
    'clip_len'    : 16,
    'batch_size'  : 4,
    'save_freq'   : 10,
    'lr'          : 1e-3,
    'useTest'     : True
    'device'      : torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
}

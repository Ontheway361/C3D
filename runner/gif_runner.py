#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/04/03
author: lujie
"""

import os
import cv2
import torch
import imageio
import numpy as np
from IPython import embed
from network import C3D_model
from utils.path_utils import PathSet

# torch.backends.cudnn.benchmark = True   # speed for single video


class Gif_generator(object):

    def __init__(self, epoch_id, dataset = 'ucf101', cls_id = 2, video_id = 1):

        self.epoch_id  = epoch_id
        self.dataset   = dataset
        self.cls_id    = cls_id
        self.video_id  = video_id
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _get_videoinfo(self):
        ''' Get the classname and video name '''

        class_names = []
        with open(PathSet.label_dir(), 'r') as f:
            class_names = f.readlines()
            f.close()

        target_class = class_names[self.cls_id-1].split(' ')[-1].strip()

        files_list = []
        _, dir = PathSet.db_dir(database = self.dataset)
        for file in os.listdir(os.path.join(dir, 'test', target_class)):
            files_list.append(file)

        if len(files_list) < self.video_id:
            self.video_id = len(files_list)

        video_name = files_list[self.video_id-1]

        return class_names, target_class, video_name

    @staticmethod
    def center_crop(frame, size = (112, 112)):
        ''' Crop center of the frame '''

        h, w = np.shape(frame)[0:2]
        th, tw = size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        frame = frame[y1:y1 + th, x1:x1 + tw, :]
        return np.array(frame).astype(np.uint8)


    def gif_generator(self):
        '''
        Generate a demo for show

        step - 1. prepare the target  video
        step - 2. load the arrchitecture and epoch_model
        step - 3. start inference
        '''

        # step - 1
        class_list, class_name, video_name = self._get_videoinfo()
        video_path = os.path.join(PathSet.root_dir(), 'dataset/ucf101_related/UCF-101', class_name, video_name + '.avi')
        video = cv2.VideoCapture(video_path)

        # step - 2
        model = C3D_model.C3D(num_classes=101).to(self.device)
        model_path = PathSet.model_dir('C3D', self.epoch_id)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # step - 3
        retaining, clip, text_imglist = True, [], []
        while retaining:

            retaining, frame = video.read()
            if not retaining and frame is None:
                continue

            tmp_ = self.center_crop(cv2.resize(frame, (171, 128)), size=(112, 112))
            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])  # normalize
            clip.append(tmp)

            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(self.device)
                with torch.no_grad():
                    outputs = model.forward(inputs)

                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

                cv2.putText(frame, class_list[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1)
                text_imglist.append(frame)
                clip.pop(0)

            # cv2.imshow('test video', frame)
            # cv2.waitKey(2)

        gif_path = os.path.join(PathSet.root_dir(), 'model_demo/C3D/', class_name + video_name + '.gif')
        imageio.mimsave(gif_path, text_imglist, fps=12)
        video.release()
        # cv2.destroyAllWindows()

#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/03/30
author: lujie
"""

import os
import glob
import torch
import timeit
import socket
from tqdm import tqdm
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from IPython import embed
from network import C3D_model
from utils.path_utils import PathSet
from utils.data_utils import VideoDataset


class C3D_Train(object):

    def __init__(self, params_dict = {}):
        ''' get the params from params_dict '''

        self.dataset      = params_dict.pop('dataset', 'ucf101')
        self.num_classes  = params_dict.pop('num_classes', 101)
        self.model_name   = params_dict.pop('model_name', 'C3D')
        self.pretrained   = params_dict.pop('pretrained', True)
        self.num_epochs   = params_dict.pop('num_epochs', 100)
        self.resume_epoch = params_dict.pop('resume_epoch', 0)
        self.frame_mode   = params_dict.pop('frame_mode', 0)   # 0 : continuous  |  1 : uniform intervals
        self.clip_len     = params_dict.pop('clip_len', 16)
        self.batch_size   = params_dict.pop('batch_size', 4)    # 20
        self.save_freq    = params_dict.pop('save_freq', 20)
        self.lr           = params_dict.pop('lr', 1e-3)
        self.useTest      = params_dict.pop('useTest', True)
        self.device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _pre_model(self):
        ''' Prepare the model '''

        model, train_params = None, None

        if self.model_name == 'C3D':
            model = C3D_model.C3D(num_classes=self.num_classes, pretrained=self.pretrained)
            train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': self.lr},
                            {'params': C3D_model.get_10x_lr_params(model), 'lr': self.lr * 10}]
        else:
            raise TypeError('Unknown model name ...')

        model.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        optimizer = optim.SGD(train_params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        if self.resume_epoch == 0:
            print("Training {} from scratch...".format(self.model_name))
        else:
            resume_file = PathSet.model_dir(model_name = self.model_name, cur_epochs =self.resume_epoch)

            checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage)

            print("Initializing weights from: {}...".format(resume_file.split('/')[-1]))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])

        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        #model.to(self.device); criterion.to(self.device)

        model_cache = (model, criterion, optimizer, scheduler)

        return model_cache


    def _pre_data(self):
        ''' load the data for train and test '''

        print('Training model on {} dataset...'.format(self.dataset))
        train_dataloader = DataLoader(VideoDataset(dataset=self.dataset, split='train',clip_len=self.clip_len, \
                           frame_mode = self.frame_mode), batch_size=self.batch_size, shuffle=True, num_workers=4)

        val_dataloader   = DataLoader(VideoDataset(dataset=self.dataset, split='val',  clip_len=self.clip_len, \
                           frame_mode = self.frame_mode), batch_size=self.batch_size, num_workers=4)

        test_dataloader  = DataLoader(VideoDataset(dataset=self.dataset, split='test', clip_len=self.clip_len, \
                           frame_mode = self.frame_mode), batch_size=self.batch_size, num_workers=4)


        trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
        data_cache = (trainval_loaders, test_dataloader)

        return data_cache


    def model_train(self):
        ''' train the C3D model according ot params_dict '''

        # step - 1
        model, criterion, optimizer, scheduler = self._pre_model()

        # step - 2
        trainval_loaders, test_dataloader = self._pre_data()
        trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
        test_size = len(test_dataloader.dataset)
        tainval_sizes, test_size = 0, 0
        log_dir = PathSet.log_dir()
        writer = SummaryWriter(log_dir=log_dir)

        # step - 3
        for epoch in range(self.resume_epoch, self.num_epochs):

                for phase in ['train', 'val']:

                    if phase == 'train':
                        print('%s\nepoch_info : (%3d|%3d)\n%s' % ('-'* 100, epoch+1, self.num_epochs, '-'*100))

                    start_time = timeit.default_timer()
                    running_loss, running_corrects = 0.0, 0.0

                    if phase == 'train':
                        scheduler.step()
                        model.train()
                    else:
                        model.eval()

                    for inputs, labels in tqdm(trainval_loaders[phase]):

                        # move inputs and labels to the device the training is taking place on
                        inputs = Variable(inputs, requires_grad=True).to(self.device)
                        labels = Variable(labels).to(self.device)
                        optimizer.zero_grad()            # optimizer.zero_grad()

                        if phase == 'train':
                            outputs = model(inputs)
                        else:
                            with torch.no_grad():
                                outputs = model(inputs)

                        probs = nn.Softmax(dim=1)(outputs)
                        preds = torch.max(probs, 1)[1]
                        loss  = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / trainval_sizes[phase]
                    epoch_acc = running_corrects.double() / trainval_sizes[phase]

                    if phase == 'train':
                        writer.add_scalar('monitor/train_loss_epoch', epoch_loss, epoch)
                        writer.add_scalar('monitor/train_acc_epoch', epoch_acc, epoch)
                    else:
                        writer.add_scalar('monitor/val_loss_epoch', epoch_loss, epoch)
                        writer.add_scalar('monitor/val_acc_epoch', epoch_acc, epoch)

                    stop_time = timeit.default_timer()
                    exe_time = stop_time - start_time
                    print("[%s]\texe_time:%.2f\tloss:%.4f\tacc:%.4f" % (phase, exe_time, epoch_loss, epoch_acc))


                if (epoch + 1) % self.save_freq == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        }, PathSet.model_dir(self.model_name, epoch + 1))
                    print("Save model at {}\n".format(PathSet.model_dir(self.model_name, epoch + 1)))

                if self.useTest:
                    self.model_infer(model, test_dataloader, writer, epoch)

        writer.close()


    def model_infer(self, model, dataloader, writer = None, cur_epochs = 0, mode = 'train'):
        '''
        Use model to infer on var/test-set

        Args :
            model      : to be tested model
            writer     : used in train-test mode to record the details
            cur_epochs : order-index of current epochs
            mode       : if model is train means model infer during the train process
        '''

        model.eval()
        start_time = timeit.default_timer()

        criterion = nn.CrossEntropyLoss()
        test_size = len(dataloader.dataset)
        running_loss, running_corrects = 0.0, 0.0

        for inputs, labels in tqdm(dataloader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss  = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size

        if mode == 'train':
            writer.add_scalar('monitor/test_loss_epoch', epoch_loss, cur_epochs)
            writer.add_scalar('monitor/test_acc_epoch', epoch_acc, cur_epochs)

        stop_time = timeit.default_timer()
        exe_time = stop_time - start_time
        print("[%s]\texe_time:%.2f\tloss:%.4f\tacc:%.4f" % ('test', exe_time, epoch_loss, epoch_acc))


if __name__ == "__main__":

    C3D_engine = C3D_Train()
    C3D_engine.model_train()

#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2020/03/06
author: lujie
"""

import torch
import torch.nn as nn
from network import C3D
from thop import profile


if __name__ == "__main__":

    inputs = torch.rand(16, 3, 16, 112, 112)
    net = C3D(num_classes=2, pretrained=False)
    net.eval()
    # flops, params = profile(net, inputs)
    outputs = net.forward(inputs)
    print(outputs.size())

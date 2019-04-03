#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/04/03
author: lujie
"""


from runner.gif_runner import Gif_generator


if __name__ == '__main__':

    gif_engine = Gif_generator(epoch_id=15, cls_id=80, video_id=3)

    gif_engine.gif_generator()

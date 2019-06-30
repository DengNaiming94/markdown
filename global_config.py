#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置全局变量
"""
from easydict import EasyDict as edict

__C = edict()

# 使用者可以通过"global_config.cfg"来获取全局变量
cfg = __C

# 训练阶段变量
__C.TRAIN = edict()

# 设置训练迭代次数
__C.TRAIN.EPOCHS = 40010
# 设置训练日志显示步长（训练n步后显示一次训练日志）
__C.TRAIN.DISPLAY_STEP = 1
# 设置验证日志显示步长（训练n步后显示一次验证日志）
__C.TRAIN.TEST_DISPLAY_STEP = 1000
# 设置训练优化器动量系数
__C.TRAIN.MOMENTUM = 0.9
# 设置训练初始学习率
__C.TRAIN.LEARNING_RATE = 0.0005
# 设置训练可用GPU内存百分占比
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# 设置训练过程中GPU性能可否超过默认值
__C.TRAIN.TF_ALLOW_GROWTH = True
# 设置训练的批量大小
__C.TRAIN.BATCH_SIZE = 4
# 设置验证的批量大小
__C.TRAIN.VAL_BATCH_SIZE = 4
# 设置学习率下降的迭代次数
__C.TRAIN.LR_DECAY_STEPS = 100000
# 设置学习率下降的速率
__C.TRAIN.LR_DECAY_RATE = 0.1
# 设置训练的图片高度
__C.TRAIN.IMG_HEIGHT = 256
# 设置训练的图片宽度
__C.TRAIN.IMG_WIDTH = 512

# 测试阶段变量
__C.TEST = edict()

# 设置测试可用GPU内存百分占比
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# 设置测试过程中GPU性能可否超过默认值
__C.TEST.TF_ALLOW_GROWTH = True
# 设置测试的批量大小
__C.TEST.BATCH_SIZE = 4

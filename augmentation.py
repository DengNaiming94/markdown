#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集增强(包含HSV格式随机扰动，随机Gramma变换以及随机高斯模糊三个增强操作)
"""
import os
import os.path as ops
import argparse
import time
import glob
import glog as log
import numpy as np 
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

class Augmentation(object):
    '''
    创建数据增强工具类
    '''

    def __init__(self):
        pass

    @staticmethod
    def _hsv_transform(image, hue_delta, sat_mult, val_mult):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float)
        image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_delta) % 180
        image_hsv[:, :, 1] *= sat_mult
        image_hsv[:, :, 2] *= val_mult
        image_hsv[image_hsv > 255] = 255
        return cv2.cvtColor(np.round(image_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 对图像进行HSV格式的随机扰动，hue_vari, sat_vari, val_vari分别为在色调、饱和度以及亮度三个通道的随机扰动幅度
    def random_hsv_transform(self, image, hue_vari, sat_vari, val_vari):
        hue_delta = np.random.uniform(-hue_vari, hue_vari)
        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        return self._hsv_transform(image, hue_delta, sat_mult, val_mult)

    @staticmethod
    def _gamma_transform(image, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(image, gamma_table)

    # 对图像进行随机Gramma变换，gamma_vari为Gramma的扰动幅度
    def random_gamma_transform(self, image, gamma_vari):
        log_gamma_vari = np.log(gamma_vari)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        gamma = np.exp(alpha)
        return self._gamma_transform(image, gamma)

    @staticmethod
    def _gaussian_transform(image, kernel_size, sigma):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 对图像进行随机高斯模糊，2*kernel_size_ub+1为卷积核尺寸上界, sigma_ub为高斯方差上界
    def random_gaussian_transform(self, image, kernel_size_ub, sigma_ub):
        kernel_size = 2 * np.random.randint(0, kernel_size_ub) + 1
        sigma = np.random.uniform(0, sigma_ub)
        return self._gaussian_transform(image, kernel_size, sigma)

def init_args():

    parser = argparse.ArgumentParser()

    # image_dir表示待数据增强的图像目录，p_*表示是否采用*类增强变换的概率
    parser.add_argument('--image_dir', type=str, help='The src image save dir')
    parser.add_argument('--p_hsv', type=float, help='Probability to change hsv of an image', default=0.5)
    parser.add_argument('--p_gamma', type=float, help='Probability to change gamma of an image', default=0.5)
    parser.add_argument('--p_gaussian', type=float, help='Probability to change gaussian of an image', default=0.5)
    parser.add_argument('--hue_vari', type=float, help='Variation of hue', default=10.0)
    parser.add_argument('--sat_vari', type=float, help='Variation of saturation', default=0.1)
    parser.add_argument('--val_vari', type=float, help='Variation of value', default=0.1)
    parser.add_argument('--gamma_vari', type=float, help='Variation of gamma', default=2.0)
    parser.add_argument('--kernel_size_ub', type=int, help='Upbound of kernel size', default=3)
    parser.add_argument('--sigma_ub', type=float, help='Upbound of sigma', default=10.0)

    return parser.parse_args()

# 对image_dir目录下的jpg, png, jpeg的所有图像进行增强，以aug-前缀命名保存在image_dir下
def batch_random_transform(image_dir, p_hsv, p_gamma, p_gaussian, \
                           hue_vari, sat_vari, val_vari, gamma_vari, \
                           kernel_size_ub, sigma_ub):

    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    time_start = time.time()

    # 将所有图像文件进行读取
    log.info('Reading images...')
    image_path_list = glob.glob('{:s}/*.jpg'.format(image_dir), recursive=False) + \
                      glob.glob('{:s}/*.png'.format(image_dir), recursive=False) + \
                      glob.glob('{:s}/*.jpeg'.format(image_dir), recursive=False)

    image_nums = len(image_path_list)
    # 目录image_dir下不存在图像，则直接返回
    if image_nums == 0:
        log.info('No image in image_dir!')
        return
    hsv_nums = 0
    gamma_nums = 0
    gaussian_nums = 0


    data_augmentation = Augmentation()

    for image_path in image_path_list:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # 判断是否进行HSV格式的随机扰动变换
        if np.random.random() < p_hsv:
            image = data_augmentation.random_hsv_transform(image, hue_vari, sat_vari, val_vari)
            hsv_nums += 1
        # 判断是否进行随机Gramma变换
        if np.random.random() < p_gamma:
            image = data_augmentation.random_gamma_transform(image, gamma_vari)
            gamma_nums += 1
        # 判断是否进行随机高斯模糊变换
        if np.random.random() < p_gaussian:
            image = data_augmentation.random_gaussian_transform(image, kernel_size_ub, sigma_ub)
            gaussian_nums += 1

        # 以aug-为前缀对增强后的图像进行保存
        image_name = ops.split(image_path)[1]
        augm_image_name = 'aug-' + image_name
        augm_image_path = ops.join(image_dir, augm_image_name)
        cv2.imwrite(augm_image_path, image)

    log.info('Images: {:d} hsv_ratio: {:.2%} gamma_ratio: {:.2%} gaussian_ratio: {:.2%}'.format(\
        image_nums, (hsv_nums/image_nums), (gamma_nums/image_nums), (gaussian_nums/image_nums)))
    log.info('Augmentation is completed! cost_time {:.5f}s'.format(time.time()-time_start))

if __name__ == '__main__':
    # 初始化命令行参数列表
    args = init_args()

    batch_random_transform(args.image_dir, args.p_hsv, args.p_gamma, args.p_gaussian, args.hue_vari, \
        args.sat_vari, args.val_vari, args.gamma_vari, args.kernel_size_ub, args.sigma_ub)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实现LaneNet的数据解析类
"""
import os.path as ops
import cv2
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass

class DataSet(object):
    """
    创建数据集类DataSet
    """

    def __init__(self, dataset_info_file):
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
        # 设置数据集已调用next_batch方法的次数
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        # gt_img_list, gt_label_binary_list, gt_label_instance_list分别代表原图列表、二值分割标签图列表、实例分割标签图列表
        # 列表中的元素为字符串，表示图片所在位置
        gt_img_list = []
        gt_label_binary_list = []
        gt_label_instance_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                # dataset_info_file中每一行代表原图、二值分割标签图、实例分割标签图的文件位置，以空格为分割
                info_tmp = _info.strip(' ').split()
                gt_img_list.append(info_tmp[0])
                gt_label_binary_list.append(info_tmp[1])
                gt_label_instance_list.append(info_tmp[2])

        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    # 将dataset_info_file的数据顺序打乱
    def _random_dataset(self):
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list), \
        'gt_img, gt_labels_instance, gt_labels_instance的数量不匹配'

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_binary_list = []
        new_gt_label_instance_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_binary_list.append(self._gt_label_binary_list[index])
            new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_binary_list = new_gt_label_binary_list
        self._gt_label_instance_list = new_gt_label_instance_list

    # return: 大小为batch_size的三个列表，分别存储cv2格式的gt_img, gt_labels_instance, gt_labels_instance
    def next_batch(self, batch_size):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) == len(self._gt_img_list), \
        'gt_img, gt_labels_instance, gt_labels_instance的数量不匹配'

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        # 如果取到数据集末尾，则重新打乱顺序，再从头开始取
        if idx_end > len(self._gt_label_binary_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
            gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels_binary = []
            gt_labels_instance = []

            for gt_img_path in gt_img_list:
                gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))

            for gt_label_path in gt_label_binary_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
                label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
                label_binary[idx] = 1
                gt_labels_binary.append(label_binary)

            for gt_label_path in gt_label_instance_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
                gt_labels_instance.append(label_img)

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels_binary, gt_labels_instance

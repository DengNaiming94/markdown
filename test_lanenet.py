#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math
import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
# 应用VGG模型的三通道均值
VGG_MEAN = [103.939, 116.779, 123.68]

# 初始化命令行参数列表
def init_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=8)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def test_lanenet(image_path, weights_path, use_gpu):

    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    # 将原图保存为image_vis，并resize成分辨率512x256
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image - VGG_MEAN

    # Tensorflow的创建Graph过程
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    # 实例化LaneNet网络
    net = lanenet_merge_model.LaneNet(phase=phase_tensor)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    # 实例化聚类对象以及后处理对象
    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # 设置会话Session的全局配置
    if use_gpu:
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': 0})
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # Tensorflow的打开Session过程
    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        # 对原图进行二值分割以及实例分割
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('Predict a single image: cost_time {:.5f}s'.format(t_cost))

        # 对掩模结果进行聚类以及后处理
        t_start = time.time()
        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        t_cluster = time.time() - t_start
        log.info('Cluster a single image: cost_time {:.5f}s'.format(t_cluster))

        # 显示原图src_image和预测掩模结果图make_image
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.show()

    # 关闭会话Session
    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir):

    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    # 读取image_dir目录下的所有图片
    log.info('Reading images...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    if use_gpu:
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': 0})
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):

            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] Predict {:d} images: total_cost_time {:.5f}s mean_cost_time {:.5f}s'.format(
                epoch+1, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                # 批量保存预测结果图
                mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                image_name = ops.split(image_path_epoch[index])[1]
                image_save_path = ops.join(save_dir, image_name)
                cv2.imwrite(image_save_path, mask_image)

            log.info('[Epoch:{:d}] Cluster {:d} images: total_cost_time {:.5f}s mean_cost_time {:.5f}'.format(
                epoch+1, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':

    args = init_args()

    # 创建保存测试结果图像目录
    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.info('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # 单张图像的测试
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # 批量图像的测试
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)

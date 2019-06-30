#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练LaneNet模型
"""
import argparse
import math
import os
import os.path as ops
import time
import cv2
import glog as log
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor

CFG = global_config.cfg
# 应用VGG模型的三通道均值
VGG_MEAN = [103.939, 116.779, 123.68]

# 初始化命令行参数列表
def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    return parser.parse_args()

def train_net(dataset_dir, weights_path=None):

    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file), '{:s}　不存在'.format(train_dataset_file)
    assert ops.exists(val_dataset_file), '{:s}　不存在'.format(val_dataset_file)

    # 创建训练集和验证集实例train_dataset，val_dataset
    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    # Tensorflow的创建Graph过程
    with tf.device('/gpu:0'):
        # input_tensor：输入张量，binary_label_tensor：二值分割标签，instance_label_tensor：实例分割标签，phase：训练（测试）阶段
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                             CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                             shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                    CFG.TRAIN.IMG_WIDTH, 1],
                                             name='binary_input_label')
        instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                      CFG.TRAIN.IMG_WIDTH],
                                               name='instance_input_label')
        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

        # 创建LaneNet网络架构
        net = lanenet_merge_model.LaneNet(phase=phase)

        # 计算损失
        compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,
                                       instance_label=instance_label_tensor, name='lanenet_model')
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']

        # 计算准确度
        out_logits = compute_ret['binary_seg_logits']
        out_logits = tf.nn.softmax(logits=out_logits)
        out_logits_out = tf.argmax(out_logits, axis=-1)
        out = tf.argmax(out_logits, axis=-1)
        out = tf.expand_dims(out, axis=-1)
        idx = tf.where(tf.equal(binary_label_tensor, 1))
        pix_cls_ret = tf.gather_nd(out, idx)
        accuracy = tf.count_nonzero(pix_cls_ret)
        accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

        # 设置训练迭代步数，学习率以及优化器
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   CFG.TRAIN.LR_DECAY_STEPS, CFG.TRAIN.LR_DECAY_RATE, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=CFG.TRAIN.MOMENTUM).minimize(loss=total_loss,
                                                                    var_list=tf.trainable_variables(),
                                                                    global_step=global_step)
    # 设置Tensorflow的Saver，用以保存Model
    saver = tf.train.Saver()
    # 设置Model的保存目录
    model_save_dir = 'model'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # 设置Model的名称（以训练开始时间为后缀）
    model_name = 'lanenet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # 设置Tensorflow的Summary，用以保存tboard
    # 设置tboard的目录（以训练开始时间为后缀）
    tboard_save_path = 'tboard/lanenet_{:s}'.format(str(train_start_time))
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar,
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    # 设置Session的全局配置
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # 设置训练阶段的全局参数，并打印出来
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    # Tensorflow的打开Session过程
    with sess.as_default():
        # 将Graph的信息保存在lanenet_model.pb文件中
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        # 如果不存在预训练的模型，则初始化参数从头开始训练，否则加载预训练模型进行迁移学习
        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        train_cost_time_mean = []
        val_cost_time_mean = []

        for epoch in range(train_epochs):
            # 训练部分
            t_start = time.time()

            with tf.device('/cpu:0'):
                # gt_imgs代表原图，binary_gt_labels代表二值分割标签，instance_gt_labels代表实例分割标签
                gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
                gt_imgs = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs]

                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

                binary_gt_labels = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp,
                                               interpolation=cv2.INTER_NEAREST)
                                    for tmp in binary_gt_labels]

                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
                instance_gt_labels = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels]
            phase_train = 'train'

            # 训练LaneNet网络
            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})

            # 异常处理：当损失不为数字时，打印异常并保存当前结果
            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('Epoch: {:d} Total cost: {:}'.format(epoch+1, c))
                log.error('Epoch: {:d} Total binary cost: {:}'.format(epoch+1, binary_loss))
                log.error('Epoch: {:d} Total instance cost: {:}'.format(epoch+1, instance_loss))
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                return

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            # tboard记录训练日志
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # 验证部分
            with tf.device('/cpu:0'):
                gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                    = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
                gt_imgs_val = [cv2.resize(tmp,
                                          dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                          dst=tmp,
                                          interpolation=cv2.INTER_LINEAR)
                               for tmp in gt_imgs_val]
                gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
                binary_gt_labels_val = [cv2.resize(tmp,
                                                   dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                   dst=tmp)
                                        for tmp in binary_gt_labels_val]
                binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
                instance_gt_labels_val = [cv2.resize(tmp,
                                                     dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                     dst=tmp,
                                                     interpolation=cv2.INTER_NEAREST)
                                          for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            # 验证LaneNet网络
            c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})
            
            # tboard记录验证日志
            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            # 打印训练日志
            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss, instance_loss, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            # 打印验证日志
            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            # 保存Model
            if epoch % 2000 == 0 and epoch != 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    # 关闭Session
    sess.close()

    return


if __name__ == '__main__':

    args = init_args()
    train_net(args.dataset_dir, args.weights_path)

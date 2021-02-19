#Lane Detection v1.0.0

'''
@Lane Detection
@Author : Kuldeep
@Reference: https://github.com/MaybeShewill-CV/lanenet-lane-detection
@Improvement: 1. All Lane Detection Improvement
              2. All Lane Extrapolation  
              3. Ego Lane Extraction
              4. Ego Lane Tracking
              5. Ego Path Plotting
'''

import sys
sys.path.append('./LaneNet/') #For modular lane detection section

import os
import pandas as pd

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import operator 

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

'''
@Lane_Detection
To get all lanes of an image and extraction of ego lane from them with lane extraction
'''

class Lane_Detection:
    
    def __init__(self, weights_path):
        """
        Constructor to initialize neural network weights
        :weights_path: The path of weight file for lane detection
        :return:
        """
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        # Set sess configuration
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        self.sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        self.sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=self.sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            self.variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            self.variables_to_restore = self.variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(self.variables_to_restore)

        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=weights_path)
            
    
    def init_args(self):
        """

        :return:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
        parser.add_argument('--weights_path', type=str, help='The model weights path')

        return parser.parse_args()


    def args_str2bool(self, arg_value):
        """

        :param arg_value:
        :return:
        """
        if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True

        elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr


    def test_lanenet(self, image_path):
        """

        :param image_path:
        :param weights_path:
        :return:
        """
        assert os.path.exists(image_path), '{:s} not exist'.format(image_path)

#         LOG.info('Start reading image and preprocessing')
        t_start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
#         LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

        t_start = time.time()
        loop_times = 1
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
#         LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result, lane_dict = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )

        mask_image = postprocess_result['mask_image']

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = self.minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

#         plt.figure('mask_image')
#         plt.imshow(mask_image[:, :, (2, 1, 0)])
#         plt.figure('src_image')
#         plt.imshow(image_vis[:, :, (2, 1, 0)])
#         plt.figure('instance_image')
#         plt.imshow(embedding_image[:, :, (2, 1, 0)])
#         plt.figure('binary_image')
#         plt.imshow(binary_seg_image[0] * 255, cmap='gray')
#         plt.show()
        
        return lane_dict
            
    def close_sess(self):
        """
        To close a current session
        :return:
        """
        self.sess.close()
        
    def extrapolate_lanes(self, all_lane_dict):
        """
        To extrapolate all lanes for missing area
        :all_lane_dict: The dictionary containing all lanes for an image
        :return: (y_list, all_x_list) 
        """
        start_limit_y = 240
        end_limit_y = 720
        start_limit_x = 0
        end_limit_x = 1280
        y_list = []
        all_x_list = []
        for lane_dict in all_lane_dict:
            previous_y = -1
            previous_x = -1
            y_list = []
            x_list = []
            for y in range(start_limit_y, end_limit_y):
                previous_y = y
                if y in lane_dict.keys():
                    previous_x = lane_dict[previous_y]
                elif previous_x != -1:
                    for temp_y in range(previous_y, end_limit_y):
                        if temp_y in lane_dict.keys():
                            diff_x = lane_dict[temp_y] - previous_x
                            diff_y = temp_y - previous_y + 1
                            step = diff_x / diff_y
                            previous_x = int(previous_x + step)
                            break
                    else:
                        if len(y_list) > 100:
                            diff_x = previous_x - x_list[len(x_list) - 51]
                            diff_y = 50
                            step = diff_x / diff_y
                            previous_x = int(previous_x + step)
                        else:
                            previous_x = -1
                if previous_x < start_limit_x:
                    previous_x = -1
                elif previous_x >= end_limit_x:
                    previous_x = -1
                y_list.append(previous_y)
                x_list.append(previous_x)
            all_x_list.append(x_list)
        return (y_list, all_x_list)

    def ego_lane_extraction(self, y_list, all_x_list):
        """
        To extract ego lanes from all lanes
        :y_all: The list of y for all lane
        :all_x_list: The list of x values for all lane
        :return: (left_ego_lane_x, right_ego_lane_x)  
        """
        ego_lane_numbers = {}
        start_limit_x = 0
        end_limit_x = 1280
        mid_x = (start_limit_x + end_limit_x) // 2
        left_ego_lane_no = -1
        right_ego_lane_no = -1
        for y in y_list[:len(y_list)//2:-1]:
            ego_lane_numbers = {}
            left_ego_lane_no = -1
            right_ego_lane_no = -1
            lane_count = 0
            for x_list_no in range(len(all_x_list)):
                if all_x_list[x_list_no][y_list.index(y)] > -1:
                    ego_lane_numbers[x_list_no] = all_x_list[x_list_no][y_list.index(y)]
                    lane_count += 1
                else:
                    ego_lane_numbers[x_list_no] = -1
            if lane_count >= 2:
                ego_lane_numbers = dict(sorted(ego_lane_numbers.items(), key=operator.itemgetter(1)))
                for ego_lane_no in ego_lane_numbers:
                    if ego_lane_numbers[ego_lane_no] == -1:
                        continue
                    elif ego_lane_numbers[ego_lane_no] < mid_x:
                        left_ego_lane_no = ego_lane_no
                    elif ego_lane_numbers[ego_lane_no] > mid_x:
                        right_ego_lane_no = ego_lane_no
                        break
            
            elif lane_count == 1 and right_ego_lane_no == -1 and left_ego_lane_no == -1:
                for ego_lane_no in ego_lane_numbers:
                    if ego_lane_numbers[ego_lane_no] == -1:
                        continue
                    elif ego_lane_numbers[ego_lane_no] < mid_x:
                        left_ego_lane_no = ego_lane_no
                    elif ego_lane_numbers[ego_lane_no] > mid_x:
                        right_ego_lane_no = ego_lane_no

            if right_ego_lane_no > -1 and left_ego_lane_no > -1:
                break
        if right_ego_lane_no == -1 and left_ego_lane_no == -1:
            return (-1,-1)
        elif left_ego_lane_no == -1:
            return (-1, all_x_list[right_ego_lane_no])
        elif right_ego_lane_no == -1:
            return (all_x_list[left_ego_lane_no], -1)
        return (all_x_list[left_ego_lane_no], all_x_list[right_ego_lane_no])  
    
    def plot_ego_lanes(self, img, y_list, ego_left_x_list, ego_right_x_list):
        """
        To plot ego lanes on an image
        :img:
        :y_list: The list of y values for ego lanes
        :ego_left_x_list: The list of x values for left ego lane
        :ego_right_x_list: The list of x values for right ego lane
        :return: plotted image 
        """
        ego_left_color = [255, 0, 0]
        ego_right_color = [0, 0, 255]
        for index in range(len(y_list)):
            y = y_list[index]
            if type(ego_left_x_list) != int:
                ego_left_x = ego_left_x_list[index]
                cv2.circle(img, (ego_left_x, y), 5, ego_left_color, -1)
            if type(ego_right_x_list) != int:
                ego_right_x = ego_right_x_list[index]
                cv2.circle(img, (ego_right_x, y), 5, ego_right_color, -1)
        return img
    
    def plot_ego_path(self, img, y_list, ego_left_x_list, ego_right_x_list):
        """
        To plot ego path on an image
        :img:
        :y_list: The list of y values for ego lanes
        :ego_left_x_list: The list of x values for left ego lane
        :ego_right_x_list: The list of x values for right ego lane
        :return: plotted image 
        """
        overlay = img.copy()
        output = img.copy()
        alpha = 0.3
        for index in range(len(y_list)):
            y = y_list[index]
            ego_right_x = -1
            ego_left_x = -1
            if type(ego_left_x_list) != int:
                ego_left_x = ego_left_x_list[index]

            if type(ego_right_x_list) != int:
                ego_right_x = ego_right_x_list[index]

            if (ego_left_x != -1 and ego_right_x != -1):
                cv2.line(overlay, 
                        (ego_left_x, y), 
                        (ego_right_x, y),
                        (0, 255, 0),
                         1)

        cv2.addWeighted(overlay, 
                        alpha, 
                        output, 
                        1 - alpha,
                        0, 
                        output)
        return output
        
    def ego_lane_tracking(self, last_ego_left_x_list, last_ego_right_x_list, ego_left_x_list, ego_right_x_list):
        """
        To track ego lanes for sequential images of a video
        :img:
        :last_ego_left_x_list: The list of x values for left ego lane from previous frame
        :last_ego_right_x_list: The list of x values for right ego lane from previous frame
        :ego_left_x_list: The list of x values for left ego lane for current frame
        :ego_right_x_list: The list of x values for right ego lane for current frame
        :return: plotted image 
        """
        threshold = 5
        if type(last_ego_left_x_list) != int:
            if type(ego_left_x_list) != int:
                for index in range(len(last_ego_left_x_list)):
                    if ego_left_x_list[index] == -1:
                        ego_left_x_list[index] = last_ego_left_x_list[index]
                    elif last_ego_left_x_list[index] > -1:
                        if last_ego_left_x_list[index] - ego_left_x_list[index] > threshold:
                            ego_left_x_list[index] = last_ego_left_x_list[index] - threshold
                        elif ego_left_x_list[index] - last_ego_left_x_list[index] > threshold:
                            ego_left_x_list[index] = last_ego_left_x_list[index] + threshold
            else:
                ego_left_x_list = last_ego_left_x_list

        if type(last_ego_right_x_list) != int:
            if type(ego_right_x_list) != int:
                for index in range(len(last_ego_right_x_list)):
                    if ego_right_x_list[index] == -1:
                        ego_right_x_list[index] = last_ego_right_x_list[index]
                    elif last_ego_right_x_list[index] > -1:
                        if last_ego_right_x_list[index] - ego_right_x_list[index] > threshold:
                            ego_right_x_list[index] = last_ego_right_x_list[index] - threshold
                        elif ego_right_x_list[index] - last_ego_right_x_list[index] > threshold:
                            ego_right_x_list[index] = last_ego_right_x_list[index] + threshold
            else:
                ego_right_x_list = last_ego_right_x_list
            
        return ego_left_x_list, ego_right_x_list
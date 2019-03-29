import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random

from gym import utils, spaces
from gym_webots.envs import webots_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from webots_ros.srv import set_float
from webots_ros.srv import set_int
from webots_ros.srv import get_bool

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

class WebotsCircuit2cPionner3atCameraNnEnv(webots_env.WebotsEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        self.motorNames = ["front_left_wheel", "front_right_wheel", "back_left_wheel", "back_right_wheel"]
        webots_env.WebotsEnv.__init__(self, "pioneer3at-circle.wbt")
        rospy.wait_for_service('/pioneer3at/supervisor/simulation_reset')
        self.reset_proxy = rospy.ServiceProxy('/pioneer3at/supervisor/simulation_reset', get_bool)
        self.step        = rospy.ServiceProxy('/pioneer3at/robot/time_step', set_int)
        self.vel_servs = []
        for motorName in self.motorNames:
            vel_serv    = rospy.ServiceProxy('/pioneer3at/' + motorName '/set_velocity', set_float)
            vel_servs.append(vel_serv)
        #enable lidar    
        enable_lidar = rospy.ServiceProxy('/pioneer3at/Sick_LMS_291/enable', set_int)
        enable_lidar(1)

        #enable camera
        enable_camera = rospy.ServiceProxy('pioneer3at/camera/enable', set_int)
        enable_camera(1)

        self.TIME_STEP = 32

        self.reward_range = (-np.inf, np.inf)

        self._seed()

        self.last50actions = [0] * 50

        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1

    def calculate_observation(self,data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''# 21 actions
        max_ang_speed = 0.3
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)'''

        front_left_wheel  = self.vel_servs[0]
        back_left_wheel   = self.vel_servs[2]
        front_right_wheel = self.vel_servs[1]
        back_right_wheel  = self.vel_servs[3]

        # 3 actions
        if action == 0: #FORWARD
            for vel_serv in self.vel_servs:
                vel_serv(0.2)            
        elif action == 1: #LEFT
            front_left_wheel(0.25)
            back_left_wheel(0.25)
            front_right_wheel(0.05)
            back_right_wheel(0.05)

        elif action == 2: #RIGHT
            front_left_wheel(0.05)
            back_left_wheel(0.05)
            front_right_wheel(0.25)
            back_right_wheel(0.25)

        rospy.wait_for_service('/pioneer3at/robot/time_step')
        try:
            self.step(TIME_STEP)
        except (rospy.ServiceException) as e:
            print ("/pioneer3at/robot/time_step service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pioneer3at/Sick_LMS_291/laser_scan/layer0', LaserScan, timeout=5)
            except:
                pass

        done = self.calculate_observation(data)

        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/pioneer3at/camera/image', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass


        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)


        '''# 21 actions
        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))

            if action_sum > 45: #L or R looping
                #print("90 percent of the last 50 actions were turns. LOOPING")
                reward = -5
        else:
            reward = -200'''


        # Add center of the track reward
        # len(data.ranges) = 100
        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        center_detour = abs(right_sum - left_sum)/5

        # 3 actions
        if not done:
            if action == 0:
                reward = 1 / float(center_detour+1)
            elif action_sum > 45: #L or R looping
                reward = -0.5
            else: #L or R no looping
                reward = 0.5 / float(center_detour+1)
        else:
            reward = -1

        #print("detour= "+str(center_detour)+" :: reward= "+str(reward)+" ::action="+str(action))

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))


        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state, reward, done, {}

        # test STACK 4
        #cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        #self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        #return self.s_t, reward, done, {} # observation, reward, done, info

    def reset(self):

        self.last50actions = [0] * 50 #used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/pioneer3at/supervisor/simulation_reset')
        try:
            #reset_proxy.call()
            self.reset_proxy(1)
        except (rospy.ServiceException) as e:
            print ("/pioneer3at/supervisor/simulation_reset service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/pioneer3at/robot/time_step')
        try:
            #resp_pause = pause.call()
            self.step(TIME_STEP)
        except (rospy.ServiceException) as e:
            print ("/pioneer3at/robot/time_step service call failed")

        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/pioneer3at/camera/image', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

        # test STACK 4
        #self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        #self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        #return self.s_t

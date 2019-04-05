import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random
import time

from gym import utils, spaces
from gym_webots.envs import webots_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from webots_ros.srv import set_float, get_float, set_int, get_bool

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
        webots_env.WebotsEnv.__init__(self, "pioneer3at-circle.wbt")

        self.motorNames = ["front_left_wheel", "front_right_wheel", "back_left_wheel", "back_right_wheel"]

        rospy.wait_for_service('/pioneer3at/supervisor/simulation_reset')
        self.reset_service = rospy.ServiceProxy('/pioneer3at/supervisor/simulation_reset', get_bool)
        self.step_service  = rospy.ServiceProxy('/pioneer3at/robot/time_step', set_int)
        self.mode_service  = rospy.ServiceProxy('/pioneer3at/supervisor/simulation_set_mode', set_int)
        self.enable_camera_service  = rospy.ServiceProxy('/pioneer3at/camera/enable', set_int)
        self.enable_lidar_service   = rospy.ServiceProxy('/pioneer3at/Sick_LMS_291/enable', set_int)
        self.vel_services = []
        self.pos_services = []

        for motorName in self.motorNames:
            #velocity services
            srv_name = '/pioneer3at/' + motorName + '/set_velocity'
            rospy.wait_for_service(srv_name)
            vel_serv = rospy.ServiceProxy(srv_name, set_float)
            self.vel_services.append(vel_serv)

            #position services
            serv_name = '/pioneer3at/' + motorName + '/set_position'
            rospy.wait_for_service(serv_name)
            pos_serv = rospy.ServiceProxy(serv_name, set_float)
            self.pos_services.append(pos_serv)                                                  

        #run webots in run mode (work good with sync and step)
        self.mode_service(1)


        self.basic_step_service = rospy.ServiceProxy('/pioneer3at/robot/get_basic_time_step', get_float)
        basic_step = self.basic_step_service(1)

        self.TIME_STEP = basic_step.value

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

    def enable_sensors(self):
        #print('enable_sensors')
        #enable camera
        responde = self.enable_camera_service(self.TIME_STEP)
        #print(responde)

        #enable lidar
        responde = self.enable_lidar_service(self.TIME_STEP)
        #print(responde)

    def enable_motors(self):
        #print('enable_motors')
        for service in self.pos_services:
            res=service(float('inf'))
            #print(res)

    def stop_robot(self):
        #print('vel_services')
        for vel_serv in self.vel_services:
            res=vel_serv(0)
            #print(res)

    def step_sim(self):
        self.step_service(self.TIME_STEP)

    def step(self, action):
        #print('step')

        front_left_wheel_service  = self.vel_services[0]
        back_left_wheel_service   = self.vel_services[2]
        front_right_wheel_service = self.vel_services[1]
        back_right_wheel_service  = self.vel_services[3]


        # 3 actions
        if action == 0: #FORWARD
            front_left_wheel_service(6)
            back_left_wheel_service(6)
            front_right_wheel_service(6)
            back_right_wheel_service(6)
        elif action == 1: #RIGTH
            front_left_wheel_service(6)
            back_left_wheel_service(6)
            front_right_wheel_service(0.3)
            back_right_wheel_service(0.3)

        elif action == 2: #LEFT
            front_left_wheel_service(0.3)
            back_left_wheel_service(0.3)
            front_right_wheel_service(6)
            back_right_wheel_service(6)

        #self.step_sim()

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pioneer3at/Sick_LMS_291/laser_scan/layer0', LaserScan, timeout=5)
            except:
                print("/pioneer3at/Sick_LMS_291/laser_scan/layer0 ERROR, retrying")
                #self.step_sim()

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
                    print("/pioneer3at/camera/image ERROR, retrying")
                    #self.step_sim()

            except:
                print("/pioneer3at/camera/image ERROR, retrying")
                #self.step_sim()


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

        done = self.calculate_observation(data)
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
        #print('reset')
        self.last50actions = [0] * 50 #used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/pioneer3at/supervisor/simulation_reset')
        self.reset_service(1)

        #wait until the simulation is reset
        # Unpause simulation to make observation 
        time.sleep(1) 
        self.enable_sensors()
        self.stop_robot()
        self.enable_motors()

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
                    print("/pioneer3at/camera/image ERROR, retrying")
                    #self.step_sim()
            except:
                print("/pioneer3at/camera/image ERROR, retrying")                                
                #self.step_sim()

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

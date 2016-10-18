#!/usr/bin/env python

#system imports
import math
import sys
import numpy
import threading
import subprocess
import os
import time

#ros imports
import roslib
import rospy
import tf
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    QuaternionStamped,
    PointStamped,
)
from std_msgs.msg import Header

#rsdk imports
import baxter_external_devices

import pointclouds

#given a string list (lst), prompts the user to select one of the option in the list.
def select_from_list(lst, allow_cancel=True):
    if allow_cancel:
        print("Select one (Esc to cancel)-")
    else:
        print("Select one-")
    for i in range(len(lst)):
        print("\t" + str(i) + ": " + str(lst[i]))
        
    while True:
        c = baxter_external_devices.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                if allow_cancel:
                    return -1
            elif c >= '0' and c <= '9':
                print("Selected " + str(lst[int(c)]))
                return int(c)
        else:
            rospy.sleep(0.001)
    
      
transform_point_listener = None  
def transform_point(point, from_frame, to_frame, verbose=False):
    if from_frame == to_frame:
        return point
        
    return_list = True
    if not isinstance(point, list):
        point = [point]
        return_list = False

    mat = numpy.zeros((len(point), 3))
    for i in range(len(point)):
        if isinstance(point[i], list) or isinstance(point[i], tuple):
            mat[i,0] = point[i][0]
            mat[i,1] = point[i][1]
            mat[i,2] = point[i][2]
        else:
            mat[i,0] = point[i].x
            mat[i,1] = point[i].y
            mat[i,2] = point[i].z
    
    mat = transform_matrix(mat, from_frame, to_frame, verbose=verbose)
    ret = []
    for row in mat:
        ret.append(Point(x=row[0], y=row[1], z=row[2]))
    
    if return_list:
        return ret
    else:
        return ret[0]
        
def transform_matrix(mat, from_frame, to_frame, verbose=False):
    if from_frame is to_frame:
        return mat
        
    global transform_point_listener
    if transform_point_listener is None:
        transform_point_listener = tf.TransformListener()
    rate = rospy.Rate(1.0)
    
    if verbose:
        print("Looking up transform from " + from_frame + " to " + to_frame + " for " + str(mat.shape) + " matrix.") 
    
    while True:
        try:
            hdr = Header(stamp=rospy.Time(0), frame_id=from_frame)
            mat44 = transform_point_listener.asMatrix(to_frame, hdr)
        except(tf.LookupException):
            if verbose:
                print("lookup excpetion")
            rate.sleep()
            continue
        except(tf.ExtrapolationException):
            if verbose:
                print("extrapolation excpetion")
            rate.sleep()
            continue
        break
    
    a = numpy.empty((mat.shape[0], 1))
    a.fill(1.0)
    a = numpy.hstack((mat, a))
    a = numpy.transpose(numpy.dot(mat44, numpy.transpose(a)))
    return numpy.delete(a, -1, 1)
        
# cloud must be a numpy recordarray.
def transform_cloud(cloud, from_frame, to_frame):
        if from_frame == to_frame:
            return cloud
            
        #filter out all x, y, or z's that are NaN
        mask = numpy.isfinite(cloud['x']) & numpy.isfinite(cloud['y']) & numpy.isfinite(cloud['z'])
        cloud = cloud[mask]
        
        cloud_array = pointclouds.get_xyz_points(cloud)
        cloud_array = transform_matrix(cloud_array, from_frame, to_frame, verbose=False)
        return pointclouds.xyz_to_cloud(cloud, cloud_array)
    
def transform_rotation(rotation, from_frame, to_frame):
    if from_frame is to_frame:
        return rotation
        
    global transform_point_listener
    if transform_point_listener is None:
        transform_point_listener = tf.TransformListener()
    rate = rospy.Rate(1.0)
    
    return_list = True
    if not isinstance(rotation, list):
        rotation = [rotation]
        return_list = False
    
    print("Looking up transform from " + from_frame + " to " + to_frame + " for " + str(len(rotation)) + " rotations.") 
    
    ret = []
    for r in rotation:
        while True:
            try:
                hdr = Header(stamp=rospy.Time(0), frame_id=from_frame)
                msg = QuaternionStamped(header=hdr, quaternion=r)
                retmsg = transform_point_listener.transformQuaternion(to_frame, msg)
            except(tf.LookupException):
                print("lookup excpetion")
                rate.sleep()
                continue
            except(tf.ExtrapolationException):
                print("extrapolation excpetion")
                rate.sleep()
                continue
            break
        ret.append(retmsg.quaternion)
    if return_list:
        return ret
    else:
        return ret[0]
    

def ros_topic_get(topic, msg_type, timeout=-1):
    val = [None]
    def callback(msg):
        val[0] = msg
    sub = rospy.Subscriber(topic, msg_type, callback)
    
    t = rospy.Time.now()
    while (timeout <= 0 or (rospy.Time.now() - t).to_sec() <= timeout) and (val[0] is None):
        rospy.sleep(0.01)
    sub.unregister()
    return val[0]
    
class RosListener():
    def __init__(self, topic, msg_type):
        self.sub = rospy.Subscriber(topic, msg_type, self.__callback)
        self.msg = None
        
    def __callback(self, msg):
        self.msg = msg
        
    def get(self, blocking=False):
        while blocking and self.msg is None:
            rospy.sleep(0.001)
        return self.msg
        
    def stop(self):
        self.sub.unregister()
        self.sub = None
        
    def __del__(self):
        if self.sub is not None:
            self.stop()
            
    def reset(self):
        self.msg = None





































        
        
        
        

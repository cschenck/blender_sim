#!/usr/bin/env python


import rospy

import baxter_interface
import baxter_external_devices

import math
import threading
import subprocess
import os
import Queue

from collections import namedtuple



global_grippers = {}
Waypoint = namedtuple("Waypoint", ["name", "ljpos", "rjpos", "lgrip", "rgrip", "vel"])


# This class represnets an ordered list of waypoints. The above 3 global variables are designed for
# use by this class. It abstracts away all the details of working specifically with Baxter and
# allows the user to simply call functions like playback, addwaypoint, or moveToWaypoint (functions
# that begin with __ are designed to be internal to the class). The only external Baxter-related
# variable to this class is the velocity variable, which must be passed to functions that edit or
# add waypoints. This should be a percent (0 - 100), but it does not represent the full velocity
# range for Baxter. For safety reasons, the maximum velocity has been set lower. If you'd like
# to change it, change the MAX_VEL variable below. Waypoints simply save the entire state of the
# robot, including both its arms and both its grippers (although it only considers the grippers
# open or closed).
class Script:

    MAX_VEL = 0.3 #keep in the range (0.0, 1.0]. Default 0.3

    def __init__(self, left_arm_enabled=True, right_arm_enabled=True):        
        global global_grippers
        self.left_arm_enabled = left_arm_enabled
        if left_arm_enabled:
            if not 'left' in global_grippers:
                global_grippers['left'] = baxter_interface.Gripper('left')
            self.left_arm = baxter_interface.Limb('left')
            self.left_grip = global_grippers['left']
            
        self.right_arm_enabled = right_arm_enabled
        if right_arm_enabled:
            if not 'right' in global_grippers:
                global_grippers['right'] = baxter_interface.Gripper('right')
            self.right_grip = global_grippers['right']
            self.right_arm = baxter_interface.Limb('right')
        
        self.cancel_move = False
        self.currently_executing_waypoint = -1
        
        self.__waypoint_counter = 0
    
        self.poss = []
        """form of poss: [list of Waypoint]"""
        
    def save(self, filename):
        f = open(filename, 'w')
        f.write("\n".join(str(w) for w in self.poss))
        f.close()
    
    def load(self, filename):
        f = open(filename, 'r')
        self.poss = [eval(s) for s in f.read().split('\n')]
        f.close()
    
    #By default plays back the entire script. Specifying a start or end index will change the
    #start or end point respectively.
    def playback(self, start_index = 0, end_index = -1):
        self.cancel_move = False
        if end_index == -1:
            end_index = len(self.poss)
        i = start_index
        while i < end_index:
            self.currently_executing_waypoint = i
            self.__move(self.poss[i])
            if self.cancel_move:
                break
            i = i + 1
        self.currently_executing_waypoint = -1
        
    def reversePlayback(self, start_index = -1, end_index = 0):
        self.cancel_move = False
        if start_index == -1:
            start_index = len(self.poss) - 1
        i = start_index
        while i >= end_index:
            self.currently_executing_waypoint = i
            self.__move(self.poss[i])
            if self.cancel_move:
                break
            i = i - 1
        self.currently_executing_waypoint = -1
        
    def moveToWaypoint(self, index):
        self.playback(start_index = index, end_index = index + 1)
        
    def __generateWaypoint(self, vel):
    
        if (not self.left_arm_enabled) or (not self.right_arm_enabled):
            raise Exception("Unable able to capture waypoint when one arm is disabled.")
    
        name = "Waypoint " + str(self.__waypoint_counter)
        ljpos = dict([(j, self.left_arm.joint_angle(j)) for j in self.left_arm.joint_names()])
        rjpos = dict([(j, self.right_arm.joint_angle(j)) for j in self.right_arm.joint_names()])
        lgrip = self.left_grip.position()
        rgrip = self.right_grip.position()
        return Waypoint(name, ljpos, rjpos, lgrip, rgrip, vel)
        
    #adds a waypoint. Requires the velocity to be specified.
    def addWaypoint(self, vel):
        self.poss.append(self.__generateWaypoint(vel))
        self.__waypoint_counter = self.__waypoint_counter + 1
        
    #Basically just saves the current state of the robot as the given index in the list with the
    #given velocity.
    def editWaypoint(self, index, vel):
        name = self.poss[index].name
        self.poss[index] = self.__generateWaypoint(vel)
        self.editWaypointName(index, name)
    
    def __move(self, wp, accuracy=1.0):
        
        rate = rospy.Rate(10.0)
        
        
        if self.left_arm_enabled:
            if wp.lgrip < 50:
                self.left_grip.close()
            else:
                self.left_grip.open()
        
        if self.right_arm_enabled:
            if wp.rgrip < 50:
                self.right_grip.close()
            else:
                self.right_grip.open()
        
        
        done = False
        if self.left_arm_enabled:
            self.left_arm.set_joint_position_speed(max(min(Script.MAX_VEL*wp.vel/100.0,1.0),0.0)) #default 0.3, range 0.0 - 1.0
        if self.right_arm_enabled:
            self.right_arm.set_joint_position_speed(max(min(Script.MAX_VEL*wp.vel/100.0,1.0),0.0)) #default 0.3, range 0.0 - 1.0
        dd = [9999.0, 9999.0, 9999.0, 9999.0, 9999.0]
        while not done:
            if self.cancel_move:
                break
        
            if self.left_arm_enabled:
                self.left_arm.set_joint_positions(wp.ljpos)
            if self.right_arm_enabled:
                self.right_arm.set_joint_positions(wp.rjpos)
            dist = 0.0
            if self.left_arm_enabled:
                for j in self.left_arm.joint_names():
                    dist = dist + (wp.ljpos[j] - self.left_arm.joint_angle(j))*(wp.ljpos[j] - self.left_arm.joint_angle(j))
            if self.right_arm_enabled:
                for j in self.right_arm.joint_names():
                    dist = dist + (wp.rjpos[j] - self.right_arm.joint_angle(j))*(wp.rjpos[j] - self.right_arm.joint_angle(j))
                
            dist = math.sqrt(dist)
            ss = 0.0
            for d in dd:
                ss = ss + math.fabs(d - dist)
            if ss/(wp.vel/100.0) < 0.1/accuracy:
                done = True
            elif rospy.is_shutdown():
                print("ROS shutdown, cancelled movement")
                done = True
            elif baxter_external_devices.getch() in ['\x1b', '\x03']:
                print("stopped")
                done = True
            else:
                dd.append(dist)
                del dd[0]
                rate.sleep()
        if self.left_arm_enabled:
            self.left_arm.set_joint_position_speed(0.3)
        if self.right_arm_enabled:
            self.right_arm.set_joint_position_speed(0.3)
        
    
    #Returns true if the robot is currently moving. Or it should at least. I'm not sure if it actually works.
    def currentMovement(self):
        return self.currently_executing_waypoint
        
    def numWaypoints(self):
        return len(self.poss)
        
    #Allows the user to change the name of a given waypoint for easier referencing (Waypoint 11 isn't very informative).
    def editWaypointName(self, index, new_name):
        self.poss[index] = self.poss[index]._replace(name=new_name)
        
    def getWaypointName(self, index):
        return self.poss[index].name
        
    def getWaypointVel(self, index):
        return self.poss[index].vel
        
    #Swaps the waypoint at index with the waypoint at index-1
    def moveWaypointUp(self, index):
        if index <= 0:
            return
        temp = self.poss[index-1]
        self.poss[index-1] = self.poss[index]
        self.poss[index] = temp
        
    #swaps the waypoint at index with the waypoint at index+1
    def moveWaypointDown(self, index):
        if index >= len(self.poss) - 1:
            return
        temp = self.poss[index+1]
        self.poss[index+1] = self.poss[index]
        self.poss[index] = temp
        
    def delWaypoint(self, index):
        del self.poss[index]
        
    #stops the robot moving (or it should at least).
    def haltMovement(self):
        self.cancel_move = True
    
    def __str__(self):
        return self.__class__.__name__

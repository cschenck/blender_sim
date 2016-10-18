#!/usr/bin/env python

#system imports
import math
import sys
import numpy
import threading
import subprocess
import os
import shutil
import subprocess
import tempfile
import time

#ros imports
try:
    from geometry_msgs.msg import Quaternion, Point
except:
    from collections import namedtuple
    Quaternion = namedtuple("Quaternion", "x y z w")
    Point = namedtuple("Point", "x y z")
    
    
def xyz_to_uv(pt, focal_length, resolution):
    return (int(round(pt.x*focal_length/pt.z + resolution[0]/2)), int(round(pt.y*focal_length/pt.z + resolution[1]/2)))
    
def uvd_to_xyz(uvd, focal_length, resolution):
    return Point(x=(uvd[0]-(resolution[0]/2.0))*uvd[2]/focal_length, y=(uvd[1]-(resolution[1]/2.0))*uvd[2]/focal_length, z=uvd[2])
    
    
def listToPoint(l):
    return Point(**dict(zip(['x', 'y', 'z'], l[0:3])))
    
def listToQuaternion(l):
    return Quaternion(**dict(zip(['x', 'y', 'z', 'w'], l[0:4])))
    
def quaternionToList(q):
    return (q.x, q.y, q.z, q.w)
    
def pointToList(p):
    return [p.x, p.y, p.z]
    
def quaternionToEuler(orien):
    return (math.atan2(2*orien.y*orien.w - 2*orien.x*orien.z, 1 - 2*orien.y*orien.y - 2*orien.z*orien.z),
            math.asin(2*orien.x*orien.y + 2*orien.z*orien.w),
            math.atan2(2*orien.x*orien.w - 2*orien.y*orien.z, 1 - 2*orien.x*orien.x - 2*orien.z*orien.z))

# This is wrong, don't know why...
#def eulerToQuaternion(euler):
#    c1 = math.cos(euler[0]/2.0)
#    c2 = math.cos(euler[1]/2.0)
#    c3 = math.cos(euler[2]/2.0)
#    s1 = math.sin(euler[0]/2.0)
#    s2 = math.sin(euler[1]/2.0)
#    s3 = math.sin(euler[2]/2.0)
#    
#    return Quaternion(x=s1*s2*c3 + c1*c2*s3, y=s1*c2*c3 + c1*s2*s3, z=c1*s2*c3 - s1*c2*s3, w=c1*c2*c3 - s1*s2*s3)

def eulerToQuaternion(euler):
    #heading, attitude, bank = euler
    #heading, bank, attitude = euler
    #attitude, heading, bank = euler
    #attitude, bank, heading = euler
    bank, heading, attitude = euler
    c1 = math.cos(heading/2.0)
    s1 = math.sin(heading/2.0)
    c2 = math.cos(attitude/2.0)
    s2 = math.sin(attitude/2.0)
    c3 = math.cos(bank/2.0)
    s3 = math.sin(bank/2.0)
    c1c2 = c1*c2
    s1s2 = s1*s2
    w = c1c2*c3 - s1s2*s3
    x = c1c2*s3 + s1s2*c3
    y = s1*c2*c3 + c1*s2*s3
    z = c1*s2*c3 - s1*c2*s3
    return Quaternion(x=x, y=y, z=z, w=w)
    
#function taken from:
#http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def quaternionMult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return [x, y, z, w]
   
#function taken from:
#http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def quaternionConjugate(q):
    x, y, z, w = q
    return [-x, -y, -z, w]
    
def quaternionNorm(q):
    x, y, z, w = q
    m = x**2 + y**2 + z**2 + w**2
    if m != 1.0:
        m = math.sqrt(m)
        x /= m
        y /= m
        z /= m
        w /= m
    return [x, y, z, w]
    
def quaternionInverse(q):
    x, y, z, w = q
    return quaternionConjugate(quaternionNorm(q))
    
# s.t. diff*q1 = q2
def quaternionDiff(q1, q2):
    return listToQuaternion(quaternionMult(quaternionToList(q1), quaternionInverse(quaternionToList(q2))))
    
def pointByQuaternion(point, quaternion):
    s = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
    if s == 0.0: #if the quaternion is 0
        return point
    q = [quaternion.x/s, quaternion.y/s, quaternion.z/s, quaternion.w/s]
    v = [point.x, point.y, point.z, 0.0]
    return listToPoint(quaternionMult(quaternionMult(q, v), quaternionConjugate(q)))
    
def rotate_and_translate_point(points, translation, rotation):
    if not type(points) is list:
        ret_list = False
        points = [points]
    else:
        ret_list = True
        
    if type(rotation) is list or type(rotation) is tuple:
        rotation = eulerToQuaternion(rotation)
    elif type(rotation) is Quaternion:
        pass
    else:
        raise Exception(type(rotation).__name__ + " is an unsupported rotation type.")

    ret = [listToPoint((numpy.array(pointToList(pointByQuaternion(p, rotation))) + numpy.array(pointToList(translation))).tolist()) for p in points]
    
    if ret_list:
        return ret
    else:
        return ret[0]
        
def rotateAboutPoint(point, rot_center, rot):
    newpoint = Point(x=point.x-rot_center.x, y=point.y-rot_center.y, z=point.z-rot_center.z)
    newpoint = pointByQuaternion(newpoint, rot)
    return Point(x=newpoint.x+rot_center.x, y=newpoint.y+rot_center.y, z=newpoint.z+rot_center.z)
    

# Code borrowed from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def quaternionFromMatrix(matrix):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                     [m01+m10,     m11-m00-m22, 0.0,         0.0],
                     [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                     [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = numpy.linalg.eigh(K)
    q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
    
    
def compute_distance(p1, p2):
    t1 = type(p1)
    if t1 is list or t1 is tuple or (t1 is numpy.ndarray and len(p1.shape) == 1):
        p1 = Point(x=p1[0], y=p1[1], z=p1[2])
    elif t1 is Point:
        pass
    else:
        raise Exception(t1.__name__ + " is not supported.")
        
    t2 = type(p2)
    if t2 is list or t2 is tuple or (t2 is numpy.ndarray and len(p2.shape) == 1):
        p2 = Point(x=p2[0], y=p2[1], z=p2[2])
    elif t2 is Point:
        pass
    else:
        raise Exception(t2.__name__ + " is not supported.")
        
    return math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2) + math.pow(p1.z - p2.z, 2))

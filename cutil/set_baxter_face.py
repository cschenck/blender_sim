import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import static_publisher as sp
import tempfile
import subprocess
import numpy as np
import os

ROBOT_SCREEN_SIZE = (1024,600)

_bridge = None
_pub = None
_cached_vid = (tempfile.NamedTemporaryFile(), "")

def _init_globals():
    global _bridge, _pub
    _bridge = CvBridge()
    _pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)

def set_baxter_face_image(fp):
    set_baxter_face(cv2.imread(fp))

def set_baxter_face_youtube(url):
    global _cached_vid
    if _cached_vid[1] != url:
        _cached_vid[0].close()
        _cached_vid = (tempfile.NamedTemporaryFile(suffix=".mp4"), url)
        os.remove(_cached_vid[0].name)
        subprocess.call(["youtube-dl", "-o", _cached_vid[0].name, url])
    set_baxter_face_video(_cached_vid[0].name)

def set_baxter_face_video(fp):
    video = cv2.VideoCapture(fp)
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    n = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if fps <= 0.0 or fps > 120:
        fps = 30.0
    rate = rospy.Rate(fps)
    #while video.isOpened():
    for i in range(n):
        ret, frame = video.read()
        if not ret:
            break
        set_baxter_face(frame)
        rate.sleep()
    video.release()

def set_baxter_face(img):

    global _bridge, _pub
    if not _bridge or not _pub:
        _init_globals()
    
    pad_axis = 1
    pad_size = ROBOT_SCREEN_SIZE[0]
    fill_axis = 0
    fill_size = ROBOT_SCREEN_SIZE[1]
    ratio = 1.0*fill_size/img.shape[fill_axis]
    if ratio*img.shape[pad_axis] > pad_size:
        pad_axis, pad_size, fill_axis, fill_size = fill_axis, fill_size, pad_axis, pad_size
        ratio = 1.0*fill_size/img.shape[fill_axis]
    img = cv2.resize(img, (int(round(img.shape[1]*ratio)), int(round(img.shape[0]*ratio))))
    shape = list(img.shape)
    shape[pad_axis] = (pad_size - img.shape[pad_axis])/2
    pad = np.zeros(shape, dtype=img.dtype)
    img = np.concatenate((pad, img, pad), axis=pad_axis)
    _pub.publish(_bridge.cv2_to_imgmsg(img, "bgr8"))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action="store", type=str, default=None)
    parser.add_argument('--youtube', action="store", type=str, default=None)
    parser.add_argument('--video', action="store", type=str, default=None)
    args = parser.parse_args()

    rospy.init_node('set_baxter_face', disable_signals=True)
    if args.image:
        for _ in range(10):
            set_baxter_face_image(args.image)
            rospy.sleep(0.1)
    elif args.video:
        while True:
            set_baxter_face_video(args.video)
            rospy.sleep(0.02)
    elif args.youtube:
        while True:
            set_baxter_face_youtube(args.youtube)
            rospy.sleep(0.02)
    rospy.signal_shutdown('Done')
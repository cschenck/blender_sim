#!/usr/bin/env python

#system imports
import os
import Queue
import threading
#ROS imports
import rosbag
import rospy
#connor code imports
import connor_util as cutil

class RosBagCapture:
    def __init__(self, filepath):
        self.__filepath = filepath
        self.__recording = False
        
    def startRecording(self, topics, msg_type):
        if os.path.isfile(self.__filepath):
            self.__bag = rosbag.Bag(self.__filepath, 'a')
        else:
            self.__bag = rosbag.Bag(self.__filepath, 'w')
            
        if type(topics) is not list and type(topics) is not tuple:
            topics = [topics]
        if type(msg_type) is not list and type(msg_type) is not tuple:
            msg_type = [msg_type]*len(topics)
            
        self.__topics = topics
        self.__recording = True
        self.__sub = []
        self.__queue = {}
        self.__manage_queue = True
        for t in topics:
            self.__queue[t] = Queue.Queue()
        
        thread = threading.Thread(target=self.__queue_manager)
        thread.daemon = True
        thread.start()

        for topic, mtype in zip(self.__topics, msg_type):
            self.__sub.append(rospy.Subscriber(topic, mtype, cutil.make_closure(self.__msg_callback, [topic])))
        
    def stopRecording(self):
        self.__recording = False
        processing = True
        while processing:
            todo = 0
            for (t, q) in self.__queue.iteritems():
                todo += q.qsize()
            if todo > 0:
                print("Waiting for %d messages to finish writing." % todo)
                rospy.sleep(1)
            else:
                processing = False
        self.__manage_queue = False
        self.__bag.flush()
        for sub in self.__sub:
            sub.unregister()
        self.__bag.close()
        #don't want these guys to hang around in memory
        self.__sub = None
        self.__bag = None
        
    def filepath(self):
        return self.__filepath
        
    def __msg_callback(self, topic, msg):
        if self.__recording:
            self.__queue[topic].put_nowait(msg)
            
    def __queue_manager(self):
        r = rospy.Rate(1000)
        while self.__manage_queue:
            for (t, q) in self.__queue.iteritems():
                try:
                    item = q.get_nowait()
                except Queue.Empty:
                    continue
                self.__bag.write(t, item)
                q.task_done()
            r.sleep()
            
            
            
            
            
            
            
            

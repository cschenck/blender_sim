#!/usr/bin/env python

#system imports
import argparse
import code
import colorsys
import math
import multiprocessing
import numpy as np
import os
import random
import re
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from io_utils import *

#ros imports
try:
    from geometry_msgs.msg import Quaternion, Point
except:
    from collections import namedtuple
    Quaternion = namedtuple("Quaternion", "x y z w")
    Point = namedtuple("Point", "x y z")

def make_closure(func, args1):
    def ret(*args2):
        args = list(args1) + list(args2)
        return func(*args)
    return ret


def get_time_ms():
    return int(round(time.time() * 1000))

def succeeded(func, exception_type=Exception):
    try:
        func()
        return True
    except exception_type:
        return False
                

        
def wrapAngle(angle):
    return ((angle + math.pi) % (2*math.pi)) - math.pi
        

    
def range_to_rgb_spectrogram(val, minVal=0.0, maxVal=1.0, rgb_max=1.0):
    val = (val - float(minVal))/(float(maxVal) - float(minVal))*4.5
    #Dark Blue (0,0,0.5) -> Light Blue (0,0,1.0) -> Cyan (0,1.0,1.0) -> Green (0,1.0,0) -> Yellow (1.0,1.0,0) -> Orange (1.0,0.5,0) -> Red (1.0,0,0)
    #                 (0,0,+0.5)             (0,+1.0,0)           (0,0,-1.0)          (+1.0,0,0)            (0,-1.0,0)
    if val <= 0.5: #dark blue -> light blue
        ret = (0.0, 0.0, 0.5 + 0.5*val/0.5)
    elif val <= 1.5: #light blue -> cyan
        ret = (0.0, val-0.5, 1.0)
    elif val <= 2.5: #cyan -> green
        ret = (0.0, 1.0, 1.0 - (val - 1.5))
    elif val <= 3.5: #green -> yellow
        ret = ((val - 2.5), 1.0, 0.0)
    elif val <= 4.5: #yellow -> red
        ret = (1.0, 1.0 - (val - 3.5), 0.0)
    return (ret[0]*rgb_max, ret[1]*rgb_max, ret[2]*rgb_max)
    
def grayscaleToHeatmap(grayscale, minVal=0.0, maxVal=1.0, rgb_max=1.0, reverse_channels=True, colors='Jet'):
    
    if colors.lower() == 'jet':
        #Dark Blue (0,0,0.5) -> Light Blue (0,0,1.0) -> Cyan (0,1.0,1.0) -> Green (0,1.0,0) -> Yellow (1.0,1.0,0) -> Orange (1.0,0.5,0) -> Red (1.0,0,0)
        #                 (0,0,+0.5)             (0,+1.0,0)           (0,0,-1.0)          (+1.0,0,0)            (0,-1.0,0)
        range_divides = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5]
        color_funcs = [lambda x : zip([0.0]*x.shape[0], [0.0]*x.shape[0], 0.5 + 0.5*x),
                       lambda x : zip([0.0]*x.shape[0], x, [1.0]*x.shape[0]),
                       lambda x : zip([0.0]*x.shape[0], [1.0]*x.shape[0], 1.0 - x),
                       lambda x : zip(x, [1.0]*x.shape[0], [0.0]*x.shape[0]),
                       lambda x : zip([1.0]*x.shape[0], 1.0 - x, [0.0]*x.shape[0])]
    elif colors.lower() == 'blue_red':
        #Dark Blue (0,0,0.5) -> Light Blue (0,0,1.0) -> Purple (1.0,0,1.0) -> Magenta (1.0,0,0.5) -> Red (1.0,0,0)
        range_divides = [0.0, 1.0, 2.0, 3.0, 4.0]
        color_funcs = [lambda x : zip([0.0]*x.shape[0], [0.0]*x.shape[0], 0.25 + 0.75*x),
                       lambda x : zip(x, [0.0]*x.shape[0], [1.0]*x.shape[0]),
                       lambda x : zip([1.0]*x.shape[0], [0.0]*x.shape[0], 1.0 - x),
                       lambda x : zip([1.0]*x.shape[0], x, [0.0]*x.shape[0])]
    elif colors.lower() == 'hsv':
        idxs = np.where(grayscale > minVal + 0.0001)
        if idxs[0].shape[0] == 0:
            eps = 0.0001
        else:
            eps = (grayscale[idxs].min() - 0.0001 - minVal)/(maxVal - minVal)
        range_divides = [0.0, eps, 1.0]
        color_funcs = [lambda x : [(0.0,0.0,0.0) for xx in x],
                       lambda x : [colorsys.hsv_to_rgb(xx, 1.0, 1.0) for xx in x]]
    else:
        raise Exception("Unrecognized color scheme %s." % colors)

    range_top = range_divides[-1] 
    grayscale = np.maximum(np.minimum((grayscale - float(minVal))/(float(maxVal) - float(minVal))*range_top, range_top), 0.0)
    heatmap = np.ones((grayscale.shape[0], grayscale.shape[1], 3))
    for i in range(len(range_divides)-1):
        bottom, top = range_divides[i], range_divides[i+1]
        if i == 0:
            pxls = np.where((grayscale >= bottom) & (grayscale <= top))
        else:
            pxls = np.where((grayscale > bottom) & (grayscale <= top))
        if pxls[0].shape[0] > 0:
            heatmap[pxls[:2]] = np.array(color_funcs[i]((grayscale[pxls]) - bottom/(top - bottom)))
    
    if reverse_channels:
        return heatmap[:,:,::-1]*rgb_max
    else:
        return heatmap*rgb_max
        

 
    
def inrange(minVal, val, maxVal):
    return max(min(val, maxVal), minVal)
    

def index(s, c, beg=0, end=None):
    if end is None:
        end = len(s)
    if type(s) == str:
        try:
            return s.index(c, beg, end)
        except ValueError:
            return -1
    else:
        return s.index(c, beg, end)

      
      
def parse_command_args(s):
    s = re.findall('(?:[^\s,"]|"(?:\\.|[^"])*")+', s)
    i = 0
    ret = dict()
    while i < len(s):
        if not s[i].startswith('--'):
            i += 1
            continue
        key = s[i][2:]
        if i+1 < len(s) and not s[i+1].startswith('--'):
            val = s[i+1]
            if (val.startswith('"') and val.endswith('"')) or (val.startswith('\"') and val.endswith('\"')) or (val.startswith("'") and val.endswith("'")) or (val.startswith("\'") and val.endswith("\'")):
                val = val[1:-1]
            ret[key] = val
            i += 2
        elif key.startswith('no_') or key.startswith('no-'):
            key = key[3:]
            ret[key] = 'False'
            i += 1
        else:
            ret[key] = 'True'
            i += 1
    return ret
      

        

def normalize_columns(mat, MAX=None, MIN=None):
    ret_array = False
    if type(mat) is list:
        mat = np.asarray(mat)
        ret_array = True
        
    ret_1d = False
    if len(mat.shape) == 1:
        m = np.zeros((1,mat.shape[0]))
        m[0] = mat
        mat = m
        ret_1d = True
    

    if MAX is None:
        MAX = np.copy(mat[0])
        for row in mat:
            for i, v in zip(range(len(row)), row):
                if v > MAX[i]:
                    MAX[i] = v
    if MIN is None:
        MIN = np.copy(mat[0])
        for row in mat:
            for i, v in zip(range(len(row)), row):
                if v < MIN[i]:
                    MIN[i] = v
                    
    ret = np.copy(mat)
    for i in range(len(MAX)):
        ret[:,i] = (mat[:,i] - MIN[i])/(MAX[i] - MIN[i])
        
    if ret_1d:
        ret = ret[0]
        
    if ret_array:
        ret = ret.tolist()
    
    return ret, MAX, MIN
    
        
        

    

    
    

    
def flattenList(l):
    all_flat = True
    flat = [True]*len(l)
    for i, ll in enumerate(l):
        if type(ll) == tuple or type(ll) == list:
            all_flat = False
            flat[i] = False
    if all_flat:
        return l
    ret = []
    for i, ll in enumerate(l):
        if flat[i]:
            ret.append(ll)
        else:
            ret += flattenList(ll)
    return ret
    

class ProgressMonitor:
    def __init__(self, progress_func, update_interval=5.0):
        self.progress_func = progress_func
        self.update_interval = update_interval
        self.start = time.time()
        self.total_time = 0
        self.running = True
        self.__pause = False
        self.pub = SingleLineUpdater()
        self.thread = threading.Thread(target=self.__monitor__, args=())
        self.thread.daemon = True
        if update_interval is not None:
            self.thread.start()
        
    def __del__(self):
        self.stop()
        
    def stop(self):
        dur = time.time() - self.start
        m, s = divmod(dur, 60)
        h, m = divmod(m, 60)
        print("\nFinished in %d:%02d:%02d." % (h, m, s))
        self.running = False
        
    def pause(self):
        self.__pause = True
        self.total_time += time.time() - self.start
        
    def resume(self):
        self.__pause = False
        self.start = time.time()

    def print_progress(self):
        dur = self.total_time + (time.time() - self.start)
        m, s = divmod(dur, 60)
        h, m = divmod(m, 60)
        percent = 100.0*self.progress_func()
        if percent > 0.0:
            dur_ = dur/(percent/100.0) - dur
        else:
            dur_ = 0.0         
        m_, s_ = divmod(dur_, 60)
        h_, m_ = divmod(m_, 60)
        self.pub.publish("Elapsed time=%d:%02d:%02d, Percent=%.2f, Estimated Time Remaining=%d:%02d:%02d" % (h, m, s, percent, h_, m_, s_))
        
    def __monitor__(self):
        percent = 0.0
        while percent < 100.0 and self.running:
            if not self.__pause:
                self.print_progress()
            time.sleep(self.update_interval)
        self.running = False
    
    
def debug_run(f, force_run=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", dest="debug", default=False)    
    args, sys.argv = parser.parse_known_args(sys.argv)
    
    if args.debug or force_run:
        frames = [None]
        def trace_func(frame, event, args):
            if event != 'exception':
                return trace_func
            else:
                if args[-1].tb_next is None:
                    frames[0] = frame
            
        sys.settrace(trace_func)
        try:
            r = f()
        except:
            exc = traceback.format_exc()
            print(exc)
            frame = frames[0]            
            global all_frames
            f = frame
            all_frames = []
            while f is not None:
                all_frames.append(f)
                f = f.f_back
            all_frames.reverse()
            global global_current_frame_idx
            global_current_frame_idx = len(all_frames) - 1
            
            def start_frame():
                global all_frames, global_current_frame_idx, frame_funcs
                f = all_frames[global_current_frame_idx]
                # evaluate commands in current namespace
                namespace = f.f_globals.copy()
                namespace.update(f.f_locals)
                namespace['lf'] = frame_funcs['list_frames']
                namespace['uf'] = frame_funcs['up_frame']
                namespace['df'] = frame_funcs['down_frame']
                namespace['sf'] = frame_funcs['set_frame']
                namespace['hf'] = frame_funcs['help_frame']
                code.interact(banner="Set frame %s" % str(f.f_code), local=namespace)
                sys.exit()
            
            def list_frames():
                global all_frames, global_current_frame_idx
                for i, f in enumerate(all_frames):
                    if i == global_current_frame_idx:
                        flair = '*'
                    else:
                        flair = ' '
                    print('%s[%d] %s' % (flair, i, str(f.f_code)))
                    
            def up_frame():
                global all_frames, global_current_frame_idx, frame_funcs
                global_current_frame_idx = max(0, global_current_frame_idx - 1)
                frame_funcs['start_frame']()
                
            def down_frame():
                global all_frames, global_current_frame_idx, frame_funcs
                global_current_frame_idx = min(len(all_frames) - 1, global_current_frame_idx + 1)
                frame_funcs['start_frame']()
                
            def set_frame(ii):
                global all_frames, global_current_frame_idx, frame_funcs
                if type(ii) is not int:
                    raise ValueError("Index must be an integer.")
                ii = all_frames.index(all_frames[ii])
                global_current_frame_idx = ii
                frame_funcs['start_frame']()

            def help_frame():
                print("Available commands:")
                print("\t%9s\tList stack frames" % 'lf()')
                print("\t%9s\tGo up a stack frame" % 'uf()')
                print("\t%9s\tGo down a stack frame" % 'df()')
                print("\t%9s\tSet stack frame" % 'sf(FRAME)')
                print("\t%9s\tPrint this help" % 'hf()')
                
            global frame_funcs
            frame_funcs = {'start_frame':start_frame, 
                           'list_frames':list_frames, 
                           'up_frame':up_frame, 
                           'down_frame':down_frame, 
                           'set_frame':set_frame,
                           'help_frame':help_frame}

            print("BEGIN POST-MORTEM")
            print("Frames:")
            list_frames()
            help_frame()
            start_frame()
    else:
        r = f()
        
    return r

def get_counts(ll):
    ret = {}
    for ii in ll:
        if ii not in ret:
            ret[ii] = 1
        else:
            ret[ii] += 1
    return sorted(ret.items(), key=lambda x : x[1])


def string_to_table(ss, header=False, row_delim='\n', col_delim=None, dtype=str, remove_empty_rows=False):
    rows = [x for x in ss.split(row_delim) if not remove_empty_rows or x.strip() != '']
    ret = [[dtype(x) for x in (rr.split() if col_delim is None else rr.split(col_delim)) if x.strip() != ''] 
                for rr in rows]
    if header:
        hdr = ret[0]
        ret = ret[1:]
        for i in range(len(ret)):
            rr = {}
            for j in range(min(len(ret[i]), len(hdr))):
                rr[hdr[j]] = ret[i][j]
            ret[i] = rr
    return ret


def histogram(*args, **kwargs):
    counts, bins = np.histogram(*args, **kwargs)
    for i in range(len(counts)):
        print("%g" % bins[i])
        print("\t%g" % counts[i])
    print("%f" % bins[-1])


def is_type(v, dtype):
    try:
        dtype(v)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def MutableNamedTuple(name, *args):
    s  = "class %s:\n" % name
    
    # Init method.
    s += "    def __init__(self"
    for arg in args:
        s += ", %s=None" % arg
    s += "):\n"
    for arg in args:
        s += "        self.%s = %s\n" % (arg, arg)

    # Pickle method.
    s += "    def __getstate__(self):\n"
    s += "        ret = {}\n"
    for arg in args:
        s += "        ret['%s'] = self.%s\n" % (arg, arg)
    s += "        return ret\n"

    #Unpickle method.
    s += "    def __setstate__(self,state):\n"
    for arg in args:
        s += "        self.%s = state['%s']\n" % (arg, arg)

    #Reduce method.
    s += "    def __reduce__(self):\n"
    s += "        return '%s'\n" % name

    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    exec(s, frame.f_globals, frame.f_locals)

def OneOffStruct(**kwargs):
    cname = 'c' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    s  = "class %s:\n" % cname
    args = kwargs.keys()

    # Init method.
    s += "    def __init__(self"
    for arg in args:
        s += ", %s" % arg
    s += "):\n"
    for arg in args:
        s += "        self.%s = %s\n" % (arg, arg)
    s += "retval = %s(**%s)" % (cname, str(kwargs))
    exec s in globals(), locals()
    return retval


def NotifyOnCompletion(func, email=None):
    all_args = sys.argv[:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--notify', action="store", type=str, dest="email", default=None)    
    args, sys.argv = parser.parse_known_args(sys.argv)

    if not email:
        email = args.email

    def send_email(subject, msg):
        fp = tempfile.NamedTemporaryFile("w")
        fp.write("Command: " + " ".join(all_args))
        fp.write("\n\n")
        fp.write("Result:\n")
        fp.write(msg)
        fp.flush()
        os.system("mail -s '%s' %s < %s" % (subject, email, fp.name))
        fp.close()

    if email:
        try:
            ret = func()
            send_email("Process completed successfully", "SUCCESS!")
            return ret
        except:
            send_email("Process errored", "ERROR:\n" + "".join(traceback.format_exception(*sys.exc_info())))
            raise
    else:
        return func()
    

def dynamic_import(module_name, default_path="", as_name=None):
    fp = findFile(module_name + ".py", defaultPath=default_path)
    if fp is None:
        raise IOError("Unable to locate %s under path %s." % (module_name, default_path))
    sys.path.append(os.path.dirname(fp))

    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    if as_name:
        exec("import %s as %s" % (module_name, as_name), frame.f_globals, frame.f_locals)
    else:
        exec("import %s" % (module_name,), frame.f_globals, frame.f_locals)
   
#better_stdout = None
#def start_multithread_stdout():
#    global better_stdout
#    better_stdout = DummyStdOut()
#    
#def stop_multithread_stdout():
#    global better_stdout
#    better_stdout.stop()

#def better_write(val):
#    global better_stdout
#    better_stdout.write(val)
#    
#def better_print(val):
#    better_write(val + "\n")
#    
#def getChar(blocking=True):
#    while True:
#        c = baxter_external_devices.getch()
#        if c:
#            return c
#    


#class DummyStdOut:
#    def __init__(self):
#        self.realStdOut = sys.stdout
#        sys.stdout = self
#        
#        self.buffer = ""
#        self.buffer_lock = threading.Lock()
#        self.spinning = True
#        self.file = open("test.txt", "w")
#        
#        self.thread = threading.Thread(target=self.__printer, args=())
#        self.thread.daemon = True
#        self.thread.start()
#        
#        
#    def __printer(self):
#        while self.spinning:
#            self.buffer_lock.acquire()
#            if len(self.buffer) > 0 and '\n' in self.buffer:
#                subprocess.call(["stty", "sane"])
#                self.realStdOut.write(self.buffer)
#                self.realStdOut.flush()
#                self.file.write(self.buffer)
#                self.file.flush()
#                self.buffer = ""
#            self.buffer_lock.release()
#            
#            
#    def write(self, val):
#        self.buffer_lock.acquire()
#        self.buffer = self.buffer + val
#        self.buffer_lock.release()
#        
#    def flush(self):
#        pass
#        
#    def stop(self):
#        self.spinning = False
#        self.file.close()
#        sys.stdout = self.realStdOut
#        self.thread.join()

def test_heatmap(args):
    import cv2
    img = np.zeros((20,300,400,1), dtype=np.float32)
    # jet = np.zeros(img.shape[:3] + (3,), dtype=np.float32)
    # blue_red = np.zeros(img.shape[:3] + (3,), dtype=np.float32)
    hsv = np.zeros(img.shape[:3] + (3,), dtype=np.float32)
    mins = 50
    maxs = 200
    c = (img.shape[1]/2, img.shape[2]/2)
    for i in range(img.shape[0]):
        print("Frame %d of %d." % (i, img.shape[0]))
        for j in range(maxs, mins-1, -1):
            v = 1.0 - 1.0*(j - mins)/(maxs - mins)
            img[i,(c[0]-j/2):(c[0]+j/2), (c[1]-j/2):(c[1]+j/2),...] = v
        # jet[i,...] = grayscaleToHeatmap(img[i,...])
        # blue_red[i,...] = grayscaleToHeatmap(img[i,...], colors='blue_red')
        hsv[i,...] = grayscaleToHeatmap(img[i,...], colors='hsv')
        ss = 1 if i < img.shape[0]/2 else -1
        c = (c[0]+3*ss, c[1]+3*ss)

    frame = 0
    while True:
        cv2.imshow("gt", img[frame,...])
        # cv2.imshow("jet", jet[frame,...])
        # cv2.imshow("blue_red", blue_red[frame,...])
        cv2.imshow("hsv", hsv[frame,...])
        cv2.waitKey(33)
        frame = (frame + 1) % img.shape[0]


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test', choices=['heatmap'], help="The test to run.")
    args = parser.parse_args()

    if args.test == 'heatmap':
        test_heatmap(args)


if __name__ == '__main__':
    test()

































        
        
        
        

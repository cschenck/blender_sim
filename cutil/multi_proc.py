#!/usr/bin/env python

#system imports
import argparse
import code
import math
import multiprocessing
import numpy as np
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from connor_util import ProgressMonitor

try:
    import psutil
except:
    pass


def kill_pid(proc_pid):
    try:
        process = psutil.Process(proc_pid)
    except:
        return
    for proc in process.get_children(recursive=True):
        try:
            proc.kill()
        except:
            pass
    try:
        process.kill()
    except:
        pass


def runCommandsInPanedTerminator(cmds, layout=None, window_size=None):
    if layout is None:
        y = int(math.floor(math.sqrt(len(cmds))))
        x = int(math.ceil(math.sqrt(len(cmds))))
        while x*y < len(cmds):
            x += 1
        layout = (x,y)
    if window_size is None:
        import Tkinter as tk
        root = tk.Tk()
        window_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
    temp_path = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_path, "terminator"))
    fp = open(os.path.join(temp_path, "terminator", "config"), "w")
    s = """
[global_config]
  suppress_multiple_term_dialog = True
[keybindings]
[profiles]
  [[default]]
    scrollback_lines = 5000
    font = Monospace 20
    background_image = None
[layouts]
  [[default]]
"""
    items = dict()
    ordered = []
    ordered.append('main_window')
    items[ordered[-1]] = dict()
    items[ordered[-1]]['type'] = 'Window'
    items[ordered[-1]]['order'] = '0'
    items[ordered[-1]]['parent'] = '\'\''
    items[ordered[-1]]['size'] = '%d, %d' % window_size
    
    spacing = tuple(window_size[i]/layout[i] for i in range(len(layout)))
    rows = [None]*layout[1]
    rows[0] = (ordered[-1], 0)
    for i in range(0, layout[1]-1):
        ordered.append('row%d' % i)
        items[ordered[-1]] = dict()
        items[ordered[-1]]['type'] = 'VPaned'
        items[ordered[-1]]['order'] = rows[i][1]
        items[ordered[-1]]['parent'] = rows[i][0]
        items[ordered[-1]]['position'] = str(int(round(spacing[1])))
        rows[i] = (ordered[-1], 0)
        rows[i+1] = (ordered[-1], 1)
    
    cols = [[None for x in range(layout[0])] for x in range(layout[1])]
    for i in range(0, layout[1]):
        cols[i][0] = rows[i]
        for j in range(0, layout[0]-1 ):
            ordered.append('row%d_col%d' % (i, j))
            items[ordered[-1]] = dict()
            items[ordered[-1]]['type'] = 'HPaned'
            items[ordered[-1]]['order'] = cols[i][j][1]
            items[ordered[-1]]['parent'] = cols[i][j][0]
            items[ordered[-1]]['position'] = str(int(round(spacing[0])))
            cols[i][j] = (ordered[-1], 0)
            cols[i][j+1] = (ordered[-1], 1)
        
    for k in range(layout[0]*layout[1]):
        i, j = k/layout[0], k%layout[0]
        ordered.append('terminal%d' % k)
        items[ordered[-1]] = dict()
        items[ordered[-1]]['type'] = 'Terminal'
        items[ordered[-1]]['order'] = cols[i][j][1]
        items[ordered[-1]]['parent'] = cols[i][j][0]
        items[ordered[-1]]['profile'] = 'default'
        if k < len(cmds):
            items[ordered[-1]]['command'] = '\'' + cmds[k].replace('\'', '\\\'') + '\''
    

    for item in ordered:
        s += "    [[[%s]]]\n" % item
        for key in items[item]:
            s += "      %s = %s\n" % (key, str(items[item][key]))
    s += "[plugins]"
    fp.write(s)
    fp.flush()
    fp.close()
    
    env = os.environ
    env['XDG_CONFIG_HOME'] = temp_path
    subprocess.call(["terminator", "-T", "\"Terminator\""], env=env)
    
    shutil.rmtree(temp_path)


def runCommandsInTabbedTerminator(cmds, labels=None):
    try:
        import Tkinter as tk
        root = tk.Tk()
        window_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
    except:
        window_size = None

    if labels is None:
        labels = cmds

    temp_path = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_path, "terminator"))
    fp = open(os.path.join(temp_path, "terminator", "config"), "w")
    s = """
[global_config]
  suppress_multiple_term_dialog = True
[keybindings]
[profiles]
  [[default]]
    scrollback_lines = 5000
    font = Monospace 20
    background_image = None
[layouts]
  [[default]]
"""
    items = dict()
    ordered = []
    ordered.append('main_window')
    items[ordered[-1]] = dict()
    items[ordered[-1]]['type'] = 'Window'
    items[ordered[-1]]['order'] = '0'
    items[ordered[-1]]['parent'] = '\'\''
    if window_size is not None:
        items[ordered[-1]]['size'] = '%d, %d' % window_size
    parent = ordered[-1]

    if len(cmds) > 1:
        ordered.append('notebook_window')
        items[ordered[-1]] = dict()
        items[ordered[-1]]['type'] = 'Notebook'
        items[ordered[-1]]['order'] = '0'
        items[ordered[-1]]['parent'] = '\'main_window\''
        items[ordered[-1]]['labels'] = ', '.join("\'" + x + "\'" for x in labels)
        parent = ordered[-1]

    for i, cmd in enumerate(cmds):
        ordered.append('terminal%d' % i)
        items[ordered[-1]] = dict()
        items[ordered[-1]]['type'] = 'Terminal'
        items[ordered[-1]]['order'] = '%04d' % i
        items[ordered[-1]]['parent'] = '\'%s\'' % parent
        items[ordered[-1]]['profile'] = 'default'
        items[ordered[-1]]['command'] = '\'' + cmd.replace('\'', '\\\'') + '\''
    

    for item in ordered:
        s += "    [[[%s]]]\n" % item
        for key in items[item]:
            s += "      %s = %s\n" % (key, str(items[item][key]))
    s += "[plugins]"
    fp.write(s)
    fp.flush()
    fp.close()
    
    env = os.environ
    env['XDG_CONFIG_HOME'] = temp_path
    subprocess.call(["terminator", "-T", "Terminator"], env=env)
    
    shutil.rmtree(temp_path)


def combine_func_and_args(func, args):
    ret = []
    for arg in args:
        ret.append((func,arg))
    return ret

def proc_init(counter, lock, fcn):
    global global_counter, global_lock
    global_counter = counter
    global_lock = lock
    if fcn:
        fcn()

def multiproc_functor(args):
    global global_counter, global_lock
    try:
        func = args[0]
        args = args[1]
        ret = func(args)
        with global_lock:
            global_counter.value += 1
        return ret
    except:
        raise Exception("\n===================\n" + "".join(traceback.format_exception(*sys.exc_info())))

def multiproc_map(func, args, nproc=multiprocessing.cpu_count()):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(nproc)
    signal.signal(signal.SIGINT, original_sigint_handler)
    res = None
    while True:
        try:
            if res is None:
                res = pool.map_async(func, args)
            ret = res.get(0.1) # Without the timeout this blocking call ignores all signals.
            pool.close()
            break
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            raise
        except multiprocessing.TimeoutError:
            pass

    pool.join()
    return ret


class MultiProcMapper:
    def __init__(self, nproc=multiprocessing.cpu_count(), init_fcn=None, use_pool=True):
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.counter = multiprocessing.Value('i', 0, lock=False)
        self.lock = multiprocessing.Lock()
        self.nproc = nproc
        if use_pool:
            self.pool = multiprocessing.Pool(nproc, initializer=proc_init, initargs=(self.counter, self.lock, init_fcn))
        else:
            self.pool = multiprocessing.pool.ThreadPool(nproc, initializer=proc_init, initargs=(self.counter, self.lock, init_fcn))
        signal.signal(signal.SIGINT, original_sigint_handler)


    def __del__(self):
        self.pool.close()

    def map(self, func, args, show_progress=False):
        res = None
        self.counter.value = 0
        if show_progress:
            pm = ProgressMonitor(lambda : 1.0*self.get_finished_count()/len(args), update_interval=1.0)
        while True:
            try:
                if res is None:
                    res = self.pool.map_async(multiproc_functor, 
                        combine_func_and_args(func, args))
                ret = res.get(0.1) # Without the timeout this blocking call ignores all signals.
                break
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                self.pool.terminate()
                if show_progress:
                    pm.stop()
                raise
            except multiprocessing.TimeoutError:
                pass
        if show_progress:
            pm.stop()   
        return ret

    def get_finished_count(self):
        return self.counter.value

    def getNProc(self):
        return self.nproc




def test(a):
    raise Exception("FALSKDJFLKAWJERFLKJ")

def main():
    mp = MultiProcMapper(nproc=1)
    mp.map(test, [1])

if __name__ == '__main__':
    main()
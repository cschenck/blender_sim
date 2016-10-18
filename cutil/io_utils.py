#!/usr/bin/env python

#system imports
import math
import glob
import sys
import numpy as np
import threading
import subprocess
import os
import shutil
import subprocess
import tempfile
import time

def getKeyboardLine():
    return raw_input(">")

def selectOptionFromList(ll):
    while True:
        for i in range(len(ll)):
            print("[%d] %s" % (i, str(ll[i])))
        s = raw_input("(\'exit\' to quit)>")
        if s[0].lower() == 'e':
            raise Exception("Program killed by user.")
        try:
            s = int(s)
        except ValueError:
            print("Not a valid number, try again.")
            continue
        try:
            ll[s]
        except IndexError:
            print("Integer out of range, try again.")
            continue
        return s
    
def getFloatValue(init_value):
    try:
        return float(getKeyboardLine())
    except ValueError:
        return init_value
        
class SingleLineUpdater:
    def __init__(self, handle=None):
        self.__lastLine = ""
        self.__linecount = 0
        self.__handle = handle
    
    def publish(self, line):
        handle = self.__handle
        if handle is None:
            handle = sys.stdout
        line = ("%7d:" % (self.__linecount)) + line
        self.__linecount = self.__linecount + 1
        handle.write(('\b' * len(self.__lastLine)) + (' ' * len(self.__lastLine)) + ('\b' * len(self.__lastLine)) + line)
        handle.flush()
        self.__lastLine = line
        
def findFile(name, defaultPath=""):
    path = os.path.join(defaultPath, name)
    if os.path.exists(path):
        return path
    
    #pick somewhere to start the search. If the default path is a real place,
    #start there, otherwise, start wherever this source file is.
    if os.path.exists(defaultPath):
        path = defaultPath
    else:        
        path = os.path.dirname(os.path.abspath(__file__))
    
    def find(_name, _path):
        for root, dirs, files in os.walk(_path):
            if _name in files or _name in dirs:
                return os.path.join(root, _name)
        return None
    
    return find(name, path)


def globFile(name, defaultPath=""):
    path = os.path.join(defaultPath, name)
    if os.path.exists(path):
        return path
    
    #pick somewhere to start the search. If the default path is a real place,
    #start there, otherwise, start wherever this source file is.
    if os.path.exists(defaultPath):
        path = defaultPath
    else:        
        path = os.path.dirname(os.path.abspath(__file__))
    
    def find(_name, _path):
        ret = []
        for root, dirs, files in os.walk(_path):
            ret += glob.glob(os.path.join(root, _name))
        return ret
    
    return find(name, path)

    # ret = None
    # while ret is None:
    #     ret = find(name, path)
    #     if ret is None:
    #         old_path = path
    #         path = os.path.dirname(path) #go up a directory
    #         #if we are at the root directory
    #         if old_path == path:
    #             return None
    # return ret
    
def write_filehandle(fp, lines):
    for line in lines:
        fp.write(line)
    fp.flush()

def write_file(filepath, lines):
    fp = open(filepath, "w")
    write_filehandle(fp, lines)
    fp.close()

def write_csv(file_object, ar):
    lines = []
    for row in ar:
        lines.append(','.join([str(x) for x in row]) + '\n')
    write_filehandle(file_object, lines)

    
    
# Credit to effbot.org/librarybook/code.htm for loading variables into current namespace
def keyboard(banner=None):
    import code, sys

    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)

    code.interact(banner=banner, local=namespace)
    
    
def makedirs(dd):
    if os.path.exists(dd) and not os.path.isfile(dd):
        return
    elif os.path.exists(dd) and os.path.isfile(dd):
        raise ValueError("%s already exists and it is a file, not a directory." % dd)
    elif os.path.dirname(dd) == dd:
        raise ValueError("Cannot create file %s because it has no parent directory." % dd)
    makedirs(os.path.dirname(dd))
    os.mkdir(dd)
    
def file_basename(dd):
    if dd.endswith('/'):
        dd = dd[:-1]
    return os.path.basename(dd)
    
    
def find_file_by_index(fp, i):
    for j in range(1,20):
        if os.path.exists(fp % (j, i)):
            return fp % (j, i)
    raise Exception("No form of '%s' with i=%d could be found." % (fp, i))

def file_dirnames(fp):
    if fp.endswith('/'):
        fp = fp[:-1]
    dd = os.path.dirname(fp)
    if dd == fp:
        return [dd]
    else:
        return file_dirnames(dd) + [fp]

def file_join(*args):
    return os.path.join(*[args[i][1:] if i > 0 and args[i].startswith('/') else args[i] for i in range(len(args))])





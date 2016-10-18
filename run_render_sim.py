#!/usr/bin/env python

import cutil
import dist_computing.run_jobs as dist_computing
import math
import numpy as np
import os
import Queue
import random
import shutil
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', action="store", dest="hosts_file", type=str,
        help='The list of hosts to run the sim on. One per line of the form "username@host".')
    parser.add_argument('--local', action="store_true", dest="local", default=False,
        help='Run the sim on the local machine only, no ssh\'ing. Useful for debugging.')
    args, uk = parser.parse_known_args()

    if args.local:
        dist_computing.local_jobs(init)
    else:
        dist_computing.manage_jobs(args.hosts_file, init)
        
# The following details how to set each of the global variables
# to render previously simulated fluids:
#  -INPUT_DATA_DIR - The directory where the fluid sim files were
#                    saved.
#  -OUTPUT_DATA_DIR - The directory to save the output of the
#                     renders. This directory will contain one
#                     folder for each simulation, and in each of
#                     those folders will be one folder for each
#                     render of that simulation.
#  -RENDER_PREFIX - The prefix to append to each render to make it 
#                   easier to identify them later on just by name.
#                   The actual name of each render will be this prefix
#                   followed by the list of arguments used to render
#                   it.
#  -SIMULATIONS - The list of names of the fluid sims to render.
#  -MAX_NUM_RENDERS - The maximum number of renders. If this is greater
#                     than the number of possible renders that can be
#                     done, then all are rendered. Otherwise this many
#                     renders will be randomly sampled.
#  -ARGS - A dictionary mapping each render argument to all its possible
#          values. This is used when randomly sampling renders to do.
#          They will be passed to the render process as --NAME VALUE.
#  -ORDERED_ARGS - An ordered list specifying an ordering for the ARGS.
#                  This is useful when creating the names for each render.

INPUT_DATA_DIR = '/media/schenckc/lightbringer/fluid_sim_output/simulations'
OUTPUT_DATA_DIR = '/media/schenckc/lightbringer/fluid_sim_output/renders'
RENDER_PREFIX = 'render_'
SIMULATIONS = ['scene' + str(x) for x in (range(1,28) + range(31,58) + range(61,88))]
MAX_NUM_RENDERS = 10,000

ORDERED_ARGS = ['camera_loc','background_rotation','camera_pitch','camera_dis','bowl_mat_preset','cup_mat_preset',
                'water_reflect','water_ior','bool_render_water']
ARGS = {'camera_loc' : ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'],
        'background_rotation' : ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'],
        'camera_pitch' : ['high', 'low'],
        'camera_dis' : ['close', 'normal', 'far'],
        'bowl_mat_preset' : [str(x) for x in range(7)],
        'cup_mat_preset' : [str(x) for x in range(7)],
        'water_reflect' : ['0.0', '0.2'],
        'water_ior' : ['1.1', '1.2', '1.33'],
        'bool_render_water' : ['True', 'False'],
       }

def construct_job(render):
    job = {}
    job['code_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulator')
    job['exec'] = 'simulate_pouring.py'
    job['args'] = [x.strip() for x in render[1].split()]
    job['dep_dirs'] = [cutil.findFile("fluid_sim_deps", cutil.file_dirnames(os.path.abspath(__file__))[-2]), cutil.__path__[0]]
    job['dep_env_vars'] = ['PYTHONPATH', 'BLENDERPATH']
    job['input_dir'] = os.path.join(INPUT_DATA_DIR, render[0].split('/')[0])
    job['output_dir'] = os.path.join(OUTPUT_DATA_DIR, render[0])
    return job

def deconstruct_job(job):
    prefix = os.path.join(cutil.file_basename(os.path.dirname(job['output_dir'])), cutil.file_basename(job['output_dir']))
    args = ' '.join([x for x in job['args'] if x != '--render'])
    return (prefix, args)

def init():

    ret = SpecialQueue()

    # Make the list of all possible renders.
    print("Computing all possible renders. This may take a minute...")
    def rec_possible_renders(i):
        if i >= len(ORDERED_ARGS):
            return [('', '--render --track_cup_lip')]
        rr = []
        sub = rec_possible_renders(i+1)
        for opt in ARGS[ORDERED_ARGS[i]]:
            for prefix, arg_list in sub:
                rr.append((opt + "_" + prefix, "--%s %s %s" % (ORDERED_ARGS[i], opt, arg_list)))
        return rr

    all_possible = [(RENDER_PREFIX + x, y) for x,y in rec_possible_renders(0)]
    print("Found %d possible renders." % (len(all_possible)*len(SIMULATIONS)))

    # Next figure out which ones are done already.
    print("Reading directory '%s' for finished renders. This may take a few minutes..." % OUTPUT_DATA_DIR)
    finished = {scene:set() for scene in SIMULATIONS}
    for scene in [x for x in os.listdir(INPUT_DATA_DIR) if x in SIMULATIONS]:
        for render in [x for x in os.listdir(os.path.join(OUTPUT_DATA_DIR, scene)) if x.startswith(RENDER_PREFIX)]:
            finished[scene].add(render)
    nfinished = reduce(lambda x,y : x+y, [len(ll) for ll in finished.values()])
    nunfinished = len(all_possible)*len(SIMULATIONS) - nfinished
    print("Found %d finished renders, %d unfinished." % (nfinished, nunfinished))

    # Render all possible renders if its less than the MAX_NUM_RENDERS, otherwise sample.
    max_nrenders = min(MAX_NUM_RENDERS, len(all_possible)*len(SIMULATIONS))

    def sample_thread():
        ss = set(all_possible)
        todo = {scene:ss.difference(finished[scene]) for scene in SIMULATIONS}
        counts = {scene:len(finished[scene]) for scene in SIMULATIONS}
        for i in range(to_queue):
            scene = None
            for ss, cc in counts.items():
                if scene is None or cc < counts[scene]:
                    scene = ss
            vv = random.sample(todo[scene], 1)[0]
            todo[scene].remove(vv)
            ret.special_put((os.path.join(scene, vv[0]), vv[1]))
            counts[scene] += 1
            time.sleep(0.01)

    thread = threading.Thread(target=sample_thread)
    thread.daemon = True
    thread.start()
    # Sleep for 3s to allows the queue to build up a bit.
    time.sleep(3)
    return ret

# Special queue class that allows adding to the front or back of the queue.
class SpecialQueue:
    def __init__(self):
        self.count = 0
        self.head = None
        self.tail = None
        self.lock = threading.Lock()
    def get(self):
        while self.empty(release_on_empty=False):
            time.sleep(0.01)
        old_head = self.head
        self.head = old_head['next']
        if self.head is not None:
            self.head['previous'] = None
        else:
            self.tail = None
        self.count -= 1
        ret = construct_job(old_head['data'])
        self.lock.release()
        print("Returning job " + str(old_head['data'][0]))
        return ret
    def put(self, job):
        to_put = deconstruct_job(job)
        self.lock.acquire()
        old_head = self.head
        self.head = {'previous':None, 'next':old_head, 'data':to_put}
        if old_head is not None:
            old_head['previous'] = self.head
        else:
            self.tail = self.head
        self.count += 1
        self.lock.release()
    def special_put(self, job):
        self.lock.acquire()
        old_tail = self.tail
        self.tail = {'next':None, 'previous':old_tail, 'data':job}
        if old_tail is not None:
            old_tail['next'] = self.tail
        else:
            self.head = self.tail
        self.count += 1
        self.lock.release()
    def empty(self, release_on_empty=True):
        self.lock.acquire()
        ret = self.head == None
        if release_on_empty:
            self.lock.release()
        return ret
    def qsize(self):
        return self.count






if __name__ == '__main__':
    main()

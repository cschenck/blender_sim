#!/usr/bin/env python

import argparse
import cutil
import dist_computing
import os
import Queue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', action="store", dest="hosts_file", type=str,
        help='The list of hosts to run the sim on. One per line of the form "username@host".')
    parser.add_argument('--local', action="store_true", dest="local", default=False,
        help='Run the sim on the local machine only, no ssh\'ing. Useful for debugging.')
    args, uk = parser.parse_known_args()

    if args.local:
        dist_computing.local_jobs(os.path.abspath(__file__))
    else:
        dist_computing.manage_jobs(args.hosts_file, os.path.abspath(__file__))
        
# The following shows how to setup the list of all simulations to run.
# Each simulation needs a name and a list of the simulation arguments.
# The global variable JOBS is a list of tuples, where the first element
# is the simulation name and the second is a string containing all the
# command line args for the simulation. The function init() below combines
# JOBS with other parameters to create a list of simulations to be run.
# The other argument to be set is DATA_DIR, which just specifies where
# to save the results of the simulations.

DATA_DIR = '/media/schenckc/lightbringer/fluid_sim_output/simulations'
JOBS = """
scene1:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup
scene2:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup
scene3:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup
scene4:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup
scene5:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup
scene6:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup
scene7:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup
scene8:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup
scene9:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup
scene10:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle
scene11:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle
scene12:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle
scene13:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle
scene14:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle
scene15:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle
scene16:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle
scene17:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle
scene18:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle
scene19:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug
scene20:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug
scene21:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug
scene22:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug
scene23:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug
scene24:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug
scene25:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug
scene26:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug
scene27:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug
scene31:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup --bowl_object dish
scene32:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup  --bowl_object dish
scene33:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup  --bowl_object dish
scene34:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object dish
scene35:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object dish
scene36:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object dish
scene37:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object dish
scene38:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object dish
scene39:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object dish
scene40:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object dish
scene41:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object dish
scene42:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object dish
scene43:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object dish
scene44:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object dish
scene45:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object dish
scene46:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object dish
scene47:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object dish
scene48:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object dish
scene49:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object dish
scene50:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object dish
scene51:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object dish
scene52:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object dish
scene53:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object dish
scene54:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object dish
scene55:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object dish
scene56:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object dish
scene57:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object dish
scene61:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup --bowl_object fruit
scene62:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup  --bowl_object fruit
scene63:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object cup  --bowl_object fruit
scene64:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object fruit
scene65:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object fruit
scene66:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object cup  --bowl_object fruit
scene67:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object fruit
scene68:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object fruit
scene69:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object cup  --bowl_object fruit
scene70:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object fruit
scene71:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object fruit
scene72:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object bottle  --bowl_object fruit
scene73:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object fruit
scene74:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object fruit
scene75:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object bottle  --bowl_object fruit
scene76:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object fruit
scene77:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object fruit
scene78:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object bottle  --bowl_object fruit
scene79:--init_value 0.9 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object fruit
scene80:--init_value 0.6 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object fruit
scene81:--init_value 0.3 --pour_ang_vel 9.0 --pour_end_time 11.0 --pouring_object mug  --bowl_object fruit
scene82:--init_value 0.9 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object fruit
scene83:--init_value 0.6 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object fruit
scene84:--init_value 0.3 --pour_ang_vel 18.0 --pour_end_time 10.0 --pouring_object mug  --bowl_object fruit
scene85:--init_value 0.9 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object fruit
scene86:--init_value 0.6 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object fruit
scene87:--init_value 0.3 --pour_ang_vel 20.0 --pour_end_time 5.0 --return_ang_vel -20.0 --return_start_time 7.0  --pouring_object mug  --bowl_object fruit
"""
JOBS = [x.strip().split(':') for x in JOBS.split('\n') if len(x.strip()) > 0]

def init():
    ret = Queue.Queue()
    for name,args in JOBS:
        job = {}
        job['code_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulator')
        job['exec'] = 'simulate_pouring.py'
        job['args'] = ['--bake'] + [x.strip() for x in args.split()]
        job['dep_dirs'] = [cutil.findFile("fluid_sim_deps", cutil.file_dirnames(os.path.abspath(__file__))[-2]), cutil.__path__[0]]
        job['dep_env_vars'] = ['PYTHONPATH', 'BLENDERPATH']
        job['output_dir'] = os.path.join(DATA_DIR, name)
        print("FOUND JOB:" + str(job))
        ret.put(job)
    return ret



if __name__ == '__main__':
    main()

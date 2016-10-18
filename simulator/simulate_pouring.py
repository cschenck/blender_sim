#!/usr/bin/env python

import argparse
import cutil
import math
import os
import sys
from table_environment import TableEnvironment, AlternateBowlTableEnvironment
import tempfile

cardinal_angles = {'N': 0*math.pi/4.0, 'NW':1*math.pi/4.0, 'W':2*math.pi/4.0, 'SW':3*math.pi/4.0, 'S':4*math.pi/4.0, 'SE':-3*math.pi/4.0, 'E':-2*math.pi/4.0, 'NE':-1*math.pi/4.0}

if os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    sys.argv = sys.argv[2:]
else:
    input_dir = None
    output_dir = sys.argv[1]
    sys.argv = sys.argv[1:]

parser = argparse.ArgumentParser(sys.argv)
# Bake arguments.
parser.add_argument('--init_value', action="store", dest="init_value", type=float, default=0.5)
parser.add_argument('--sim_time', action="store", dest="sim_time", type=float, default=15.0)
parser.add_argument('--pour_ang_vel', action="store", dest="pour_ang_vel", type=float, default=9.0)
parser.add_argument('--return_ang_vel', action="store", dest="return_ang_vel", type=float, default=-9.0)
parser.add_argument('--pour_end_time', action="store", dest="pour_end_time", type=float, default=10.0)
parser.add_argument('--return_start_time', action="store", dest="return_start_time", type=float, default=15.0)
parser.add_argument('--pouring_object', action="store", dest="pouring_object", type=str, default="cup")
parser.add_argument('--bowl_object', action="store", dest="bowl_object", type=str, default="bowl")
parser.add_argument('--alternate_bowl', action="store", dest="alternate_bowl", type=str, default=None)

# Render arguments
parser.add_argument('--camera_loc', action="store", dest="camera_loc", type=lambda x : cardinal_angles[x], default='N')
parser.add_argument('--background_rotation', action="store", dest="background_rotation", type=lambda x : cardinal_angles[x], default='N')
parser.add_argument('--video_start_ratio', action="store", dest="video_start_ratio", type=float, default=-1.0) #DEPRICATED
parser.add_argument('--use_video_plane', action="store_true", dest="use_video_plane", default=True)
parser.add_argument('--composite_video_plane', action="store_true", dest="composite_video_plane", default=False)
parser.add_argument('--no_composite_video_plane', action="store_false", dest="composite_video_plane")
parser.add_argument('--render_pouring_object', action="store", dest="render_pouring_object", type=str, default=None)
parser.add_argument('--render_glass_cup', action="store_true", dest="render_glass_cup", default=False)
parser.add_argument('--render_cup', action="store_true", dest="render_cup", default=True)
parser.add_argument('--no_render_cup', action="store_false", dest="render_cup")
parser.add_argument('--camera_pitch', action="store", dest="camera_pitch", type=lambda x : {'high':math.pi/4.0,'low':0.0}[x], default='high')
parser.add_argument('--camera_dis', action="store", dest="camera_dis", type=lambda x : {'close':8.0,'normal':10.0,'far':12.0}[x], default='normal')
parser.add_argument('--bowl_mat_preset', action="store", dest="bowl_mat_preset", type=int, default=None)
parser.add_argument('--cup_mat_preset', action="store", dest="cup_mat_preset", type=int, default=None)
parser.add_argument('--water_alpha', action="store", dest="water_alpha", type=float, default=0.0)
parser.add_argument('--water_reflect', action="store", dest="water_reflect", type=float, default=0.2)
parser.add_argument('--water_ior', action="store", dest="water_ior", type=float, default=1.33)
parser.add_argument('--render_water', action="store_true", dest="render_water")
parser.add_argument('--no_render_water', action="store_false", dest="render_water")
parser.add_argument('--bool_render_water', action="store", type=str, dest="bool_render_water")

parser.add_argument('--bake', action="store_true", dest="bake", default=False)
parser.add_argument('--render', action="store_true", dest="render", default=False)
parser.add_argument('--track_cup_lip', action="store_true", dest="track_cup_lip", default=False)
# parser.add_argument('--load', action="store_true", dest="load", default=False)
# parser.add_argument('--render_path', action="store", dest="render_path", type=str, default=None)
# parser.add_argument('--blend_file', action="store", dest="blend_file", type=str, default=None)
# parser.add_argument('--bake_path', action="store", dest="bake_path", type=str, default=None)
parser.add_argument('--data_prefix', action="store", dest="data_prefix", type=str, default="data")
parser.add_argument('--gt_prefix', action="store", dest="gt_prefix", type=str, default="ground_truth")

parser.add_argument('--dry_run', action="store_true", dest="dry_run", default=False)
parser.add_argument('--tempdir', action="store", dest="tempdir", type=str, default=None)
args = parser.parse_args()

if args.tempdir:
    tempfile.tempdir = args.tempdir

if args.bool_render_water is not None:
    args.render_water = eval(args.bool_render_water)

if input_dir is None:
    args.load = False
    args.blend_file = os.path.join(output_dir, 'scene.blend')
    args.bake_path = os.path.join(output_dir, 'bake_files')
    cutil.makedirs(args.bake_path)
    args.render_path = "/dev/null"
else:
    args.load = True
    args.blend_file = os.path.join(input_dir, 'scene.blend')
    args.bake_path = os.path.join(input_dir, 'bake_files')
    args.render_path = output_dir
    #TODO hack
    args.render = True

BLENDERPATHS = [os.path.expanduser('~')]
if 'BLENDERPATH' in os.environ:
    BLENDERPATHS += os.environ['BLENDERPATH'].split(':')
for path in BLENDERPATHS:
    if not os.path.exists(path):
        continue
    bp = cutil.findFile("blender-2.69", path)
    if bp is not None:
        blender_path = os.path.join(bp, "blender")
        break

if blender_path is None:
    raise Exception("Can't find blender executable!")

env_args = {'blender_path':blender_path, 'pouring_object':args.pouring_object, 'bowl_object':args.bowl_object}
if args.alternate_bowl is not None:
    env_args['alternate_bowl'] = args.alternate_bowl
    constructor = AlternateBowlTableEnvironment
else:
    constructor = TableEnvironment

if args.dry_run:
    print("Running as dry run.")
else:
    print("NOT running as dry run. THIS IS THE REAL DEAL!")

if args.load:
    env = constructor(filepath=args.blend_file, **env_args)
else:
    env = constructor(**env_args)
    env.fillCup(args.init_value)
    env.setDuration(args.sim_time)
    env.setCupAngle(0.0, 0.0)
    args.pour_ang_vel = args.pour_ang_vel/180.0*math.pi
    args.return_ang_vel = args.return_ang_vel/180.0*math.pi
    max_angle = args.pour_ang_vel*args.pour_end_time
    if max_angle > math.pi:
        max_angle = math.pi
        max_time = math.pi/args.pour_ang_vel
    else:
        max_time = args.pour_end_time
    if args.alternate_bowl is None:
        env.setCupAngle(max_angle, max_time)
    else:
        t = 0.0
        while t <= max_time + 0.1:
            env.setCupAngle(max_angle*t/max_time, t)
            t += 0.1

    if args.return_start_time < args.sim_time:
        env.setCupAngle(max_angle, args.return_start_time)
        min_angle = max_angle + args.return_ang_vel*(args.sim_time - args.return_start_time)
        if min_angle < 0:
            max_time = args.return_start_time + max_angle/abs(args.return_ang_vel)
            min_angle = 0.0
        else:
            max_time = args.sim_time
        if args.alternate_bowl is None:
            env.setCupAngle(min_angle, max_time)
        else:
            t = args.return_start_time
            while t <= max_time + 0.1:
                env.setCupAngle((1.0 - (t - args.return_start_time)/(max_time - args.return_start_time))*(max_angle - min_angle) + min_angle, t)
                t += 0.1

    
if args.render or args.track_cup_lip:
    if args.render_pouring_object is not None:
        replace_cup = {'cup_type':args.render_pouring_object, 'glass':args.render_glass_cup, 'surface_subsample':True}
    else:
        replace_cup = None
        
    if args.use_video_plane:
        video_angle = cutil.wrapAngle(args.camera_loc - args.background_rotation)
        video_name = min([(k, abs(cutil.wrapAngle(cardinal_angles[k] - video_angle))) for k in cardinal_angles], key=lambda x : x[1])[0]
        video_start_ratio = {'N':1.0, 'NW':0.0, 'W':1.0, 'SW':1.0, 'S':0.0, 'SE':1.0, 'E':1.0, 'NE':0.0}[video_name]
        background_video = {'filepath':os.path.join(cutil.findFile("textures", os.path.dirname(os.path.abspath(__file__))), video_name + ".avi"), 'start_ratio':video_start_ratio}
    else:
        background_video = None
    
    env.prepareRender(replace_cup=replace_cup, background_video=background_video, cup_mat_preset=args.cup_mat_preset, bowl_mat_preset=args.bowl_mat_preset, camera_angle=args.camera_loc, camera_pitch=args.camera_pitch, camera_dis=args.camera_dis, water_alpha=args.water_alpha, water_reflect=args.water_reflect, water_ior=args.water_ior)
    
    env.setCupVisible(visible=args.render_cup)
    env.rotateBackground(args.background_rotation)
    
        
if args.data_prefix == "None":
    args.data_prefix = None
if args.gt_prefix == "None":
    args.gt_prefix = None

if args.render or args.bake:
    env.generateOutput(render_path=args.render_path, 
                       bake_path=args.bake_path, 
                       render=args.render, 
                       bake=args.bake, 
                       dry_run=args.dry_run, 
                       data_prefix=args.data_prefix, 
                       gt_prefix=args.gt_prefix, 
                       composite_video_plane=args.composite_video_plane, 
                       render_water=args.render_water) 

if args.track_cup_lip:
    env.trackCupLip(args.render_path)

if args.bake:
    env.sim.saveScene(args.blend_file)
    
#env.sim.saveScene("/home/schenckc/temp/test2.blend")
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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

# General arguments used for either rendering or simulation.
parser.add_argument('--bake', action="store_true", dest="bake", default=False,
    help='Run in simulation mode (i.e., "bake" the simulation).')
parser.add_argument('--render', action="store_true", dest="render", default=False,
    help='Run in render mode.')
parser.add_argument('--track_cup_lip', action="store_true", dest="track_cup_lip", default=False,
    help='After rendering, generate a text file with the x,y pixel coordinates of the lip of the pouring object for each frame on each line.')
parser.add_argument('--dry_run', action="store_true", dest="dry_run", default=False,
    help='Don\'t actually execute any simulation or rendering. Just ensure that everything will run properly. Useful for verifying everything is setup correctly before attempting to run.')
parser.add_argument('--tempdir', action="store", dest="tempdir", type=str, default=None,
    help='The simulation and rendering processes use a significant amount of temporary files. By default, they use the default temporary file storage. Set this variable to have them use a user-specified directory for temporary files.')

# Simulation arguments. Only needed when simulating the fluid trajectory.
parser.add_argument('--init_value', action="store", dest="init_value", type=float, default=0.5,
    help='The initial ratio to fill the cup [0-1].')
parser.add_argument('--sim_time', action="store", dest="sim_time", type=float, default=15.0,
    help='The number of seconds to run the simulation for.')
parser.add_argument('--pour_end_time', action="store", dest="pour_end_time", type=float, default=10.0,
    help='The time at which to end the initial tild downward of the cup (which starts immediately).')
parser.add_argument('--pour_ang_vel', action="store", dest="pour_ang_vel", type=float, default=9.0,
    help='The angular velocity of the cup during the initial tilt down of the cup.')
parser.add_argument('--return_start_time', action="store", dest="return_start_time", type=float, default=15.0,
    help='The time to start tilting the cup back upright, which continues until the sim times out.')
parser.add_argument('--return_ang_vel', action="store", dest="return_ang_vel", type=float, default=-9.0,
    help='The angular velocity while tilting the cup back upright.')
parser.add_argument('--pouring_object', action="store", dest="pouring_object", type=str, default="cup",
    help='The source container to pour from. May be cup, bottle, or mug.')
parser.add_argument('--bowl_object', action="store", dest="bowl_object", type=str, default="bowl",
    help='The target container to pour into. May be bowl, dish, or fruit.')
parser.add_argument('--alternate_bowl', action="store", dest="alternate_bowl", type=str, default=None,
    help='DEPRICATED: Switches out the target container with a cup. Same values as for pouring_object.')

# Render arguments. Only needed when rendering a simulation.
parser.add_argument('--camera_loc', action="store", dest="camera_loc", type=lambda x : cardinal_angles[x], default='N',
    help='The azimuth of the camera around the table. Can be N, NW, W, SW, S, SE, E, NE.')
parser.add_argument('--background_rotation', action="store", dest="background_rotation", type=lambda x : cardinal_angles[x], default='N',
    help='The rotation of the background sphere. Can be N, NW, W, SW, S, SE, E, NE. If using a background video, which video to use is selected by determing what part of the background is being looked at by the camera by coming camera_loc with this variable.')
parser.add_argument('--video_start_ratio', action="store", dest="video_start_ratio", type=float, default=-1.0,
    help='DEPRICATED: don\'t use') #DEPRICATED
parser.add_argument('--use_video_plane', action="store_true", dest="use_video_plane", default=True,
    help='Add this argument to insert a background video behind the table.')
parser.add_argument('--composite_video_plane', action="store_true", dest="composite_video_plane", default=False,
    help='If set, instead of adding the background video to the render, the video is added after the render is finished. This was added to see if adding the video after the fact would be faster, but it is not and it looks worse. This option is not recommended.')
parser.add_argument('--no_composite_video_plane', action="store_false", dest="composite_video_plane",
    help='Disables --composite_video_plane.')
parser.add_argument('--render_pouring_object', action="store", dest="render_pouring_object", type=str, default=None,
    help='This can be either True or False (passed as --render_pouring_object True). If False, the pouring object is not rendered.')
parser.add_argument('--render_glass_cup', action="store_true", dest="render_glass_cup", default=False,
    help='This can be either True or False (passed as --render_glass_cup True). If true, the pouring object is rendered as glass.')
parser.add_argument('--render_cup', action="store_true", dest="render_cup", default=True,
    help='If set, the pouring object is rendered.')
parser.add_argument('--no_render_cup', action="store_false", dest="render_cup",
    help='If set, the pouring object is not rendered.')
parser.add_argument('--camera_pitch', action="store", dest="camera_pitch", type=lambda x : {'high':math.pi/4.0,'low':0.0}[x], default='high',
    help='The pitch of the camera. "high" is looking down into the target container at a 45 degree angle. "low" is looking directly at the side of the target container.')
parser.add_argument('--camera_dis', action="store", dest="camera_dis", type=lambda x : {'close':8.0,'normal':10.0,'far':12.0}[x], default='normal',
    help='The camera distance from the scene. Can be close, normal, or far.')
parser.add_argument('--bowl_mat_preset', action="store", dest="bowl_mat_preset", type=int, default=None,
    help='The target container texture preset to use. Can be 0 to 6 inclusive.')
parser.add_argument('--cup_mat_preset', action="store", dest="cup_mat_preset", type=int, default=None,
    help='The pouring object texture preset to use. Can be 0 to 6 inclusive.')
parser.add_argument('--water_alpha', action="store", dest="water_alpha", type=float, default=0.0,
    help='The transparency of the liquid, where 0 is fully transparent and 1 is fully opaque.')
parser.add_argument('--water_reflect', action="store", dest="water_reflect", type=float, default=0.2,
    help='The reflectance of the liquid. See Blender\'s liquid property settings for details.')
parser.add_argument('--water_ior', action="store", dest="water_ior", type=float, default=1.33,
    help='The index of refraction for the liquid.')
parser.add_argument('--render_water', action="store_true", dest="render_water",
    help='DEPRICATED: use --bool_render_water instead.')
parser.add_argument('--no_render_water', action="store_false", dest="render_water",
    help='DEPRICATED: use --bool_render_water instead.')
parser.add_argument('--bool_render_water', action="store", type=str, dest="bool_render_water",
    help='This can be either True or False (passed as --bool_render_water True). If false, the liquid is not shown in the render.')
parser.add_argument('--data_prefix', action="store", dest="data_prefix", type=str, default="data",
    help='The prefix for the rendered images. The files will be of the form DATA_PREFIX\%04d.png where \%04d is the frame number.')
parser.add_argument('--gt_prefix', action="store", dest="gt_prefix", type=str, default="ground_truth",
    help='The prefix for the pixel labels of the rendered images. The files will be of the form GT_PREFIX\%04d.png where \%04d is the frame number.')


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

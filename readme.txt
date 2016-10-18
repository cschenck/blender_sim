This is the readme for a library setup to use Blender to automatically generate sequences of RGB and image labels of liquids pouring in Blender's built-in fluid simulator. This library is still under development. Please send any questions to connor.schenck@gmail.com.

####### REQUIREMENTS ########

This library has only been tested on Ubuntu 14.04 64-bit machines, however it should work on any linux x64 machine. This library uses ssh to log in to many machines simultaneously to simulate & render. Please ensure each is running linux 64-bit. This library does not use the GPU, so no GPU is required on any machine. However, please ensure that you can ssh into each machine without using a password (e.g., use ssh-copy-id to copy your ssh key).

Please ensure to install all the dependencies listed in deps.txt on the host machine you will be running this from, however, dependencies are automatically copied to all remote machines, so you do not need to install these dependencies on them. You do not need to install Blender on any machine; this library requires a special version of Blender, which has been included with this repository.

####### CONCEPTUAL OVERVIEW #######

This library is written entirely in python 2.7.

Blender uses the convention of splitting the generation of images from a fluid simulator into two steps: simulation and rendering. This library maintains that convention. This is largely for practical reasons: simulating the trajectory of a fluid takes much longer on the CPU than rendering that simulation. By splitting this process into two steps, we can first simulate a fluid trajectory, saving the intermediate result, and then render that trajectory many times, altering properites that don't affect the trajectory of the fluid (e.g., textures, backgrounds, etc.). This allows generating a large dataset relatively quickly.

The data storage for this library is split into two folders. The first is the simulation folder, which holds all the intermediate calculations of the fluid trajectory. There is one folder for each simulation run, and in each is a Blender .blend file and a folder containing the fluid trajectory (which Blender stores seperetely from the .blend file). The second folder is a render folder, which holds the final result of the render. It contains one folder for each simulation, which each contain one folder for each render. Each render folder contains two images for each frame of the simulation: one of the raw color, and one of pixel labels for that frame (red=cup pixels, green=bowl pixels, blue=liquid pixels, alpha=which of the three is "on top").

Practically speaking, these folders take up a significant amount of storage space, so you should ensure that you have plenty before beginning.

####### HOW TO RUN A SIMULATION #######

The easiest way to run the simulator is to run run_fluid_sim.py and then run_render_sim.py. The first runs the fluid simulations and stores the intermediate result. The second uses that result to generate renders. Each has a set of variables you must set before running, which are described towards the top of each file. Each also will ssh into a set of remote machines to perform the simulations/renders (unless you pass the --local command). Please refer to each file for how to pass that list as a command line argument.

To see the list of arguments for the simulator, please refer to simulator/simulate_pouring.py.

####### FILES ####### 

Here is how the file structure of this library is layed out:

-cutil - This folder contains a library of utility functions used by this library. Most are self-explanatory (e.g., cutil.findFile(X,Y) finds file X by recursively searching directory Y).

-dist_computing - This folder contains code to automatically distribute the simulation/render computations across multiple machines via ssh. It automatically copies the code, relevant data, and dependencies to the remote machine, runs the simulation/render, and copies the result back to wherever you specified the output directory to be.

-fluid_sim_deps - This folder contains dependencies required by the fluid simulator (including a custom version of Blender). It is automatically copied to any remote machine that needs it.

-simulator - This folder contains the actual code to programmatically simulate a fluid in a pouring scenario. The files are as follows:
 |--> run_sim.py - Because Blender's internal python interpreter uses python3, but the rest of the code uses python2.7, commands are passed to Blender via a local socket connection. This file is started with Blender (i.e., 'blender --python run_sim.py'). It listens on a specific port and executes any python commands that come over that port in the Blender environment. This is done so that the rest of the library can run in python2.7.
 |--> sim_lib.py - This is the base simulator library. It handles starting up Blender in a seperate process. It then passes commands over the socket to run_sim.py to execute. The purpose of this library is to convert easy to understand function calls to Blender code (e.g., saveScene(filepath) -> bpy.ops.wm.save_as_mainfile(filepath=filepath), however there are much more complicated examples as well). Please refer to this file for a complete list of available commands. If you wish to add additional commands, you may call __exec_cmd__(cmd, ret_val) to execute a command in the Blender environment, where cmd is a python command in a string, and ret_val is true if you want the result returned, or false if not (note that in python, if you require the result from a command, it must be a single line with no newline characters).
 |--> table_environment.py - This file sets up the environment to perform pours in. It creates an instance of the class in sim_lib.py and calls it to setup the table, the bowl on the table, and the cup above the bowl. It also adds liquid to the scene, and sets the background if so desired. It also has functions for simulating and rendering the scene. While the file sim_lib.py provides access directly to Blender's internal commands, this class sits on top of that and provides more high-level access to functions required to set up a pouring scenario.
 |--> simulate_pouring.py - This class actually creates an instance of the table_environment.py class, and simulates/renders the corresponding scene. This class takes in a long list of arguments to define how the scene will be setup (refer to the file for a list and description). Only arguments required for simulating are required when generating the fluid simulation; and only arguments required for rendering are required when generating the renders. If you want to run a simulation stand-alone from the python scripts listed in the previous section, this is the file to run. This is the file executed on each remote machine when running the simulation.
 |--> settings.py - Settings related to how the run_sim.py and sim_lib.py sockets should communicate with each other.
 |--> models - This folder contains each of the bowl models used by the simulator.
 |--> textures - This folder contains the textures and background videos used during rendering.
 
-deps.txt - A list of dependencies that must be installed on the host machine (but not the remotes) to run this library.

-host_list.txt - A list of remote machines, one per line, in the form username@host. Please ensure that username@host can be logged in without a password (e.g., use ssh-copy-id to copy your ssh key).

-run_fluid_sim.py - A python script to run the fluid simulation step and save the intermediate result. Please refer to the comment towards the top of this file for how to set all the variables before running.

-run_render_sim.py - A python script for generating the render output from the fluid simulation intermediate results. Please refer to the comment towards the top of this file for how to set all the variables before running.

-readme.txt - This file :P .
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

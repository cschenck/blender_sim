#!/usr/bin/env python

import argparse
import glob
import math
import multiprocessing
import numpy as np
import os
import Queue
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time

#third-party imports
import cutil
try:
    import cv2
    cv2_imported = True
except:
    cv2_imported = False
import imageio

#local imports
import settings


DEVNULL = open(os.devnull, 'wb')


class FluidSim:
    def __init__(self, headless, run_sim, filepath, blender_path, port=None):
    
        self.blender_path = blender_path
        self.filepath = filepath
        if run_sim:
            if port is None:
                self.port = random.randint(10000, 31900)
            else:
                self.port = port
            print("Starting up simu...")
            fp = cutil.findFile("run_sim.py", os.path.dirname(os.path.realpath(__file__)))
            cmd = [blender_path]
            if headless:
                cmd.append("--background")
            cmd.append("--python")
            cmd.append(fp)
            cmd.append("--")
            if not headless:
                cmd.append("--threaded")
            cmd.append("--port")
            cmd.append(str(self.port))
            print("Starting blender with the following command:")
            print(' '.join(cmd))
            self.p = subprocess.Popen(cmd)
            
            print("Sim running.") 
        else:
            self.p = None
            self.port = settings.TCP_PORT
    
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        start = cutil.get_time_ms()
        while True:
            try:
                self.s.connect((settings.TCP_IP, self.port))
                self.connected = True
                break
            except socket.error:
                time.sleep(0.1)
#            if cutil.get_time_ms() - start > 5000:
#                raise Exception("Timed out trying to connect to simulator. Are you sure it's running?")
        print("Took " + str(cutil.get_time_ms() - start) + "ms to connect to the simulator.")
        self.s.settimeout(settings.CMD_RESPONSE_TIMEOUT)
        
        if filepath is not None:
            self.loadScene(filepath)
        else:
            self.unlinkObject("Cube")
            self.unlinkObject("Lamp")
          

    def stop(self):
        print("Shutting down sim...")
        if self.s is not None and self.connected:
            self.s.send("exit")
            time.sleep(0.5)
            self.s.close()
            self.s = None
            self.connected = False
        if self.p is not None:
            try:
                for i in range(5):
                    if self.p.poll() is None:
                        time.sleep(1)
                if self.p.poll() is None:
                    raise Exception()
                self.p.wait()
            except:
                self.p.kill()
            finally:
                self.p = None
        # Make sure Blender dies.
        try:
            import getpass
            subprocess.call(['killall', 'blender', '-u', getpass.getuser()])
        except:
            pass
        print("Shutdown.")

    def __del__(self):
        self.stop()
        
    def __exec_cmd__(self, cmd, ret_val):
        cmd = cmd.replace("\n", "#NEWLINE#")
        if ret_val:
            cmd = "eval%d:%s" % (len(cmd), cmd)
        else:
            cmd = "exec%d:%s" % (len(cmd), cmd)
        self.s.send(cmd)
        while True:
            try:
                m = self.s.recv(settings.BUFFER_SIZE)
                if m == "running":
                    time.sleep(0.01)
                elif m.startswith("success"):
                    l = int(m[7:m.index(":")])
                    r = m[(m.index(":")+1):]
                    if l - len(r) > 0:
                        m = self.s.recv(l - len(r))
                        r += m
                    return r
                elif m.startswith("error:"):
                    raise Exception(m[6:])
                else:
                    print("MALFORMED RETURN VALUE:")
                    print(m)
                    #raise Exception("Malformed return value:" + m)
            except socket.timeout, e:
                if e.args[0] == 'timed out':
                    #raise Exception("Command timed out. Maybe the server isn't running?")
                    print("Blender server timeout...")
                else:
                    raise
                    
    ##################### Misc control commands ##########################################
    
    def saveScene(self, filepath):
        self.__exec_cmd__("bpy.ops.wm.save_as_mainfile(filepath='%s')" % filepath, False)
        
    def loadScene(self, filepath):
        self.__exec_cmd__("bpy.ops.wm.open_mainfile(filepath='%s')" % filepath, False)
        
    def reloadScene(self):
        temp = tempfile.NamedTemporaryFile()
        self.saveScene(temp.name)
        self.loadScene(temp.name)
        temp.close()

    def pointCameraAt(self, loc, dis, pitch, yaw):
        x = math.cos(yaw)*dis
        y = math.sin(yaw)*dis
        z = math.sin(pitch)*dis
        self.setLocation("Camera", (x + loc[0], y + loc[1], z + loc[2]))
        D = np.array([-x, -y, -z])
        D = D/np.linalg.norm(D)
        U = np.array([0.0, 0.0, 1.0])
        S = np.cross(U, D)
        U = np.cross(D, S)
        matrix = np.zeros((3,3))
        matrix[:,0] = D
        matrix[:,1] = S
        matrix[:,2] = U
        q = cutil.quaternionToList(cutil.quaternionFromMatrix(matrix))
        r = (0.5,-0.5,-0.5,0.5) # Make the camera point along the x-axis.
        q = cutil.quaternionMult(q, r)
        self.setRotation("Camera", q)
        
    def setCameraResolution(self, res):
        self.__exec_cmd__("bpy.context.scene.render.resolution_x = %d" % res[0], False)
        self.__exec_cmd__("bpy.context.scene.render.resolution_y = %d" % res[1], False)
        self.__exec_cmd__("bpy.context.scene.render.resolution_percentage = 100", False)

    def getCameraResolution(self):
        x = eval(self.__exec_cmd__("bpy.context.scene.render.resolution_x", True))
        y = eval(self.__exec_cmd__("bpy.context.scene.render.resolution_y", True))
        return (x, y)
        
    def getCameraFOV(self):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['Camera'].data.angle", True))

    def getCameraFocalLength(self):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['Camera'].data.lens", True))
        
    def addPointLamp(self, loc, energy=1.0, distance=30.0, color=(1.0,1.0,1.0), use_diffuse=True, use_specular=True):
        s = """
def addPointLamp(loc, energy, distance, color, use_diffuse, use_specular):
    scene = bpy.context.scene
    lamps = [obj.name for obj in bpy.data.objects if obj.type == 'LAMP']
    lidx = 0
    while ('Lamp%d' % lidx) in lamps:
        lidx += 1
    lname = 'Lamp%d' % lidx
    lamp_data = bpy.data.lamps.new(name=lname, type='POINT')
    lamp_object = bpy.data.objects.new(name=lname, object_data=lamp_data)
    scene.objects.link(lamp_object)
    lamp_object.location = mathutils.Vector(loc)
    lamp_data.energy = energy
    lamp_data.distance = distance
    lamp_data.color = color
    lamp_data.use_diffuse = use_diffuse
    lamp_data.use_specular = use_specular
"""
        s += "\naddPointLamp(%s, %f, %f, %s, %s, %s)" % (str(loc), energy, distance, str(color), str(use_diffuse), str(use_specular))
        self.__exec_cmd__(s, False)
        
    def enableLights(self, enable=True):
        lamps = self.getObjects(obj_type='LAMP')
        for lamp in lamps:
            self.hideObject(lamp, hide = (not enable))
            
    def setCudaDevice(self, cudaDevice):
        self.__exec_cmd__("bpy.context.user_preferences.system.compute_device_type = 'CUDA'", False)
        self.__exec_cmd__("bpy.context.user_preferences.system.compute_device = '%s'" % cudaDevice, False)

    ######################## Commands for dealing with objects ##################################
    def getObjects(self, obj_type='MESH'):
        return eval(self.__exec_cmd__("[obj.name for obj in bpy.context.scene.objects if obj.type == '%s']" % obj_type, True))
        
    def getType(self, obj):
        return self.__exec_cmd__("bpy.context.scene.objects['%s'].type" % obj, True)
        
    def getLocation(self, obj):
        ss = self.__exec_cmd__("bpy.context.scene.objects['%s'].location" % obj, True)
        return eval(ss[ss.index("("):(ss.index(")")+1)])
        
    def getRotation(self, obj):
        ss = eval(self.__exec_cmd__("tuple([bpy.context.scene.objects['%s'].rotation_quaternion.__getitem__(i) for i in range(4)])" % obj, True))
        return tuple(list(ss[1:]) + [ss[0]])
        
    def setLocation(self, obj, loc):
        ss = self.__exec_cmd__("bpy.context.scene.objects['%s'].location = mathutils.Vector(%s)" % (obj, str(loc)), False)
        
    def setRotation(self, obj, rot):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].rotation_mode = 'QUATERNION'" % obj, False)
        rot = tuple([rot[-1]] + list(rot[:-1]))
        self.__exec_cmd__("bpy.context.scene.objects['%s'].rotation_quaternion = mathutils.Quaternion(%s)" % (obj, str(rot)), False)
        
    def addNewMeshObject(self, name, loc, verts, faces):
        self.__exec_cmd__("me = bpy.data.meshes.new('%sMesh')" % name, False)
        self.__exec_cmd__("ob = bpy.data.objects.new('%s', me)" % name, False)
        self.__exec_cmd__("ob.location = mathutils.Vector(%s)" % str(loc), False)
        self.__exec_cmd__("scn = bpy.context.scene", False)
        self.__exec_cmd__("scn.objects.link(ob)", False)
        self.__exec_cmd__("scn.objects.active = ob", False)
        self.__exec_cmd__("me.from_pydata(%s, [], %s)" % (str(verts), str(faces)), False)
        self.__exec_cmd__("me.update()", False)
        
    def setObjectSize(self, obj, size):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].dimensions = mathutils.Vector(%s)" % (obj, str(size)), False)
        
    def getObjectSize(self, obj):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].dimensions.to_tuple()" % obj, True))
        
    def hideObject(self, name, hide = True):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].hide_render = %s" % (name, str(hide)), False)
        
    def addPyramid(self, name, loc, size):
        verts = ((-size[0]/2, -size[1]/2, 0.0), 
                 (size[0]/2, -size[1]/2, 0.0),
                 (size[0]/2, size[1]/2, 0.0),
                 (-size[0]/2, size[1]/2, 0.0),
                 (0.0, 0.0, size[2]))
        faces = ((0,1,2), (2,3,0),
                 (0,4,1),
                 (1,4,2),
                 (2,4,3),
                 (3,4,0))
        self.addNewMeshObject(name, loc, verts, faces)
        
    def addCube(self, name, loc, size):
        verts = [(-size[0]/2,-size[1]/2,-size[2]/2), # 0 ll-b
                 (-size[0]/2,size[1]/2,-size[2]/2),  # 1 ul-b
                 (size[0]/2,size[1]/2,-size[2]/2),   # 2 ur-b
                 (size[0]/2,-size[1]/2,-size[2]/2),  # 3 lr-b
                 (-size[0]/2,-size[1]/2,size[2]/2),  # 4 ll-t
                 (-size[0]/2,size[1]/2,size[2]/2),   # 5 ul-t
                 (size[0]/2,size[1]/2,size[2]/2),    # 6 ur-t
                 (size[0]/2,-size[1]/2,size[2]/2)]   # 7 lr-t
        faces = [(0,1,2,3), (7,6,5,4), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
        self.addNewMeshObject(name, loc, verts, faces)
        
    def selectObject(self, obj, select=True):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].select = %s" % (obj, str(select)), False)
        
    def isSelected(self, obj):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].select" % obj, True))
        
    def unlinkObject(self, obj):
        self.__exec_cmd__("bpy.context.scene.objects.unlink(bpy.context.scene.objects['%s'])" % obj, False)
        
    def import3dsModel(self, name, filepath):
        objs = self.getObjects()
        self.__exec_cmd__("bpy.ops.import_scene.autodesk_3ds(filepath='%s')" % filepath, False)
        obj = [x for x in self.getObjects() if x not in objs][0]
        self.__exec_cmd__("bpy.context.scene.objects['%s'].name = '%s'" % (obj, name), False)
        if os.path.exists(os.path.join(os.path.dirname(filepath), "scale.txt")):
            scale = eval(open(os.path.join(os.path.dirname(filepath), "scale.txt"), "r").readlines()[0])
            self.__exec_cmd__("bpy.context.scene.objects['%s'].scale = (%f, %f, %f)" % (name, scale['x'], scale['y'], scale['z']), False)
        
    def computeBoundingBox(self, obj):
        s = """
def computeBoundingBox(obj):
    ob = bpy.context.scene.objects['Bowl']
    bb = list(zip(*[(ob.matrix_world*mathutils.Vector(corner)).to_tuple() for corner in ob.bound_box]))
    min_ = [min(dim) for dim in bb]
    max_ = [max(dim) for dim in bb]
    return (tuple(min_), tuple(max_))
"""
        self.__exec_cmd__(s, False)
        return eval(self.__exec_cmd__("computeBoundingBox('%s')" % obj, True))

    def getVertices(self, obj):
        return eval(self.__exec_cmd__("[(bpy.context.scene.objects['%s'].matrix_world*vert.co).to_tuple() for vert in bpy.context.scene.objects['%s'].data.vertices]" % (obj, obj), True))

    def getVertex(self, obj, idx):
        return eval(self.__exec_cmd__("(bpy.context.scene.objects['%s'].matrix_world*bpy.context.scene.objects['%s'].data.vertices[%d].co).to_tuple()" % (obj, obj, idx), True))

    def pointInObjectFrame(self, obj, pt):
        return eval(self.__exec_cmd__("(bpy.context.scene.objects['%s'].matrix_world.inverted()*mathutils.Vector(%s)).to_tuple()" % (obj, str(pt)), True))

    def pointToPixel(self, pt):
        ret = eval(self.__exec_cmd__("bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, bpy.context.scene.objects['Camera'], mathutils.Vector(%s)).to_tuple()[:2]" % str(pt), True))
        res = self.getCameraResolution()
        return (int(ret[0]*res[0]), int((1.0 - ret[1])*res[1]))
        
    ################## Commands to handle textures ##################################
        
    def setVideoTexture(self, obj, filepath, start_ratio):
        self.setImageTexture(obj, filepath)
        nframes = imageio.get_reader(filepath).get_length()
        anim_dur = self.getNumberFrames()
        offset = int(round(1.0*start_ratio*max(nframes - anim_dur, 0))) + 1
        s = """
def setVideoTexture(obj, offset):
    img_usr = bpy.data.objects[obj].data.materials[0].texture_slots[0].texture.image_user
    anim_dur = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    img_usr.frame_offset = offset
    img_usr.frame_duration = anim_dur
"""
        s += "setVideoTexture('%s', %d)" % (obj, offset)
        self.__exec_cmd__(s, False)
        
    def getVideoTextureSettings(self, obj):
        return self.__exec_cmd__("bpy.data.objects['%s'].data.materials[0].texture_slots[0].texture.image.name" % obj, True), eval(self.__exec_cmd__("bpy.data.objects['%s'].data.materials[0].texture_slots[0].texture.image_user.frame_offset" % obj, True))
        
    def getImageTextureSize(self, obj, specular_intensity=0.5, specular_hardness=50):
        return eval(self.__exec_cmd__("tuple(bpy.context.scene.objects['%s'].data.materials[0].texture_slots[0].texture.image.size)" % obj, True))
        
    def setImageTexture(self, obj, filepath, texture_coords = 'UV', mapping = 'FLAT', specular_intensity=0.5, specular_hardness=50):
        self.__exec_cmd__("img = bpy.data.images.load('%s')" % filepath, False)
        
        # Create image texture from image
        self.__exec_cmd__("cTex = bpy.data.textures.new('%sColorTex', type = 'IMAGE')" % obj, False)
        self.__exec_cmd__("cTex.image = img", False)
        
        # Create material
        self.__exec_cmd__("mat = bpy.data.materials.new('%sTexMat')" % obj, False)
        self.__exec_cmd__("mat.specular_intensity = %f" % specular_intensity, False)
        self.__exec_cmd__("mat.specular_hardness = %f" % specular_hardness, False)
     
        # Add texture slot for color texture
        self.__exec_cmd__("mtex = mat.texture_slots.add()", False)
        self.__exec_cmd__("mtex.texture = cTex", False)
        self.__exec_cmd__("mtex.texture_coords = '%s'" % texture_coords, False)
        self.__exec_cmd__("mtex.use_map_color_diffuse = True", False)
        self.__exec_cmd__("mtex.use_map_color_emission = True", False)
        self.__exec_cmd__("mtex.emission_color_factor = 1.0", False)
        self.__exec_cmd__("mtex.use_map_density = True", False)
        self.__exec_cmd__("mtex.mapping = '%s'" % mapping, False)
        
        self.__exec_cmd__("ob = bpy.context.scene.objects['%s']" % obj, False)
        self.__exec_cmd__("me = ob.data", False)
        self.__exec_cmd__("me.materials.append(mat)", False)
        self.__exec_cmd__("for f in me.polygons: f.material_index = len(me.materials)-1", False)
        
    def setMaterial(self, obj, diffuse_color=(0.8,0.8,0.8), diffuse_intensity=0.8, specular_intensity=0.5, specular_hardness=50):
        self.__exec_cmd__("mat = bpy.data.materials.new('%sMat')" % obj, False)
        self.__exec_cmd__("mat.diffuse_color = %s" % str(diffuse_color), False)
        self.__exec_cmd__("mat.diffuse_intensity = %f" % diffuse_intensity, False)
        self.__exec_cmd__("mat.specular_intensity = %f" % specular_intensity, False)
        self.__exec_cmd__("mat.specular_hardness = %f" % specular_hardness, False)
        self.__exec_cmd__("ob = bpy.context.scene.objects['%s']" % obj, False)
        self.__exec_cmd__("me = ob.data", False)
        self.__exec_cmd__("me.materials.append(mat)", False)
        
    def getMaterialEmit(self, obj):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].emit" % obj, True))
        
    def setMaterialEmit(self, obj, emit):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].emit = %f" % (obj, emit), False)
        
    def setMaterialShadeless(self, obj, use_shadeless=True):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].use_shadeless = %s" % (obj, str(use_shadeless)), False)
        
    def transparencyEnabled(self, obj):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].use_transparency" % obj, True))
        
    def enableTransparency(self, obj, enable=True):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].use_transparency = %s" % (obj, str(enable)), False)
        
    def mirrorEnabled(self, obj):
        return eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].raytrace_mirror.use" % obj, True))
        
    def enableMirror(self, obj, enable=True):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].data.materials[0].raytrace_mirror.use = %s" % (obj, str(enable)), False)
        
    def setFluidMaterial(self, obj, water_alpha=0.0, water_reflect=0.2, water_ior=1.33):
        self.__exec_cmd__("mat = bpy.data.materials.new('%sMat')" % obj, False)
        self.__exec_cmd__("mat.diffuse_color = (1,1,1)", False)
        self.__exec_cmd__("mat.diffuse_intensity = 0.0", False)
        self.__exec_cmd__("mat.ambient = 0.5", False)
        self.__exec_cmd__("mat.translucency = 0.5", False)
        self.__exec_cmd__("mat.use_transparency = True", False)
        self.__exec_cmd__("mat.transparency_method = 'RAYTRACE'", False)
        self.__exec_cmd__("mat.alpha = %f" % water_alpha, False)
        #self.__exec_cmd__("mat.raytrace_transparency = 1.33", False)
        self.__exec_cmd__("mat.raytrace_mirror.use = True", False)
        self.__exec_cmd__("mat.raytrace_mirror.reflect_factor = %f" % water_reflect, False)
        self.__exec_cmd__("mat.raytrace_transparency.ior = %f" % water_ior, False)
        self.__exec_cmd__("mat.raytrace_transparency.depth = 5", False)
        self.__exec_cmd__("mat.specular_hardness = 511", False)
        self.__exec_cmd__("ob = bpy.context.scene.objects['%s']" % obj, False)
        self.__exec_cmd__("me = ob.data", False)
        self.__exec_cmd__("me.materials.append(mat)", False)
        #self.__exec_cmd__("bpy.data.materials['%sMat'].raytrace_transparency.ior = 1.33" % obj, False)
        #self.__exec_cmd__("bpy.data.materials['%sMat'].raytrace_transparency.depth = 5" % obj, False)
        
    def setGlassMaterial(self, obj):
        self.__exec_cmd__("mat = bpy.data.materials.new('%sMat')" % obj, False)
        self.__exec_cmd__("mat.diffuse_color = (1,1,1)", False)
        self.__exec_cmd__("mat.diffuse_intensity = 0.1", False)
        self.__exec_cmd__("mat.use_transparency = True", False)
        self.__exec_cmd__("mat.transparency_method = 'RAYTRACE'", False)
        self.__exec_cmd__("mat.alpha = 0.1", False)
        self.__exec_cmd__("mat.raytrace_mirror.use = True", False)
        self.__exec_cmd__("mat.raytrace_mirror.reflect_factor = 0.1", False)
        self.__exec_cmd__("mat.specular_intensity = 1.0", False)
        self.__exec_cmd__("mat.specular_hardness = 511", False)
        self.__exec_cmd__("mat.translucency = 0.5", False)
        self.__exec_cmd__("mat.ambient = 0.5", False)
        self.__exec_cmd__("mat.raytrace_transparency.ior = 1.37", False)
        self.__exec_cmd__("mat.raytrace_transparency.depth = 5", False)
        self.__exec_cmd__("mat.raytrace_mirror.fresnel = 2.5", False)
        self.__exec_cmd__("ob = bpy.context.scene.objects['%s']" % obj, False)
        self.__exec_cmd__("me = ob.data", False)
        self.__exec_cmd__("me.materials.append(mat)", False)
        #self.__exec_cmd__("bpy.data.materials['%sMat'].raytrace_transparency.ior = 1.37" % obj, False)
        #self.__exec_cmd__("bpy.data.materials['%sMat'].raytrace_transparency.depth = 5" % obj, False)
        #self.__exec_cmd__("bpy.data.materials['%sMat'].raytrace_mirror.fresnel = 2.5" % obj, False)
        
    def removeMaterial(self, obj):
        self.__exec_cmd__("bpy.context.scene.objects.active = bpy.context.scene.objects['%s']" % obj, False)
        self.__exec_cmd__("bpy.ops.object.material_slot_remove()", False)
        
    def setTextureToAllFaces(self, obj):
        s = """
def setTextureToAllFaces(obj):
    bm = bmesh.new()
    bm.from_mesh(bpy.context.scene.objects[obj].data)
    uv_layer = bm.loops.layers.uv.verify()
    for f in bm.faces:
        n = len(f.loops)
        corner_idxs = [round(x*n) for x in [0.0, 0.25, 0.5, 0.75]]
        for i,corner_idx in enumerate(corner_idxs):
            f.loops[corner_idx][uv_layer].uv = mathutils.Vector((0 if i in [0,1] else 1, 0 if i in [0,3] else 1))
        for i in corner_idxs:
            j = corner_idxs[(corner_idxs.index(i) + 1) % len(corner_idxs)]
            ii = f.loops[i][uv_layer].uv
            jj = f.loops[j][uv_layer].uv
            nn = j - i + 1 if j > i else n - i + j
            for k in range(i+1,j):
                kk = 1.0*(k - i)/nn
                f.loops[k][uv_layer].uv = (jj - ii)*kk + ii
    bm.to_mesh(bpy.context.scene.objects[obj].data)
"""
        s += "\nsetTextureToAllFaces('%s')" % obj
        self.__exec_cmd__(s, False)
        
    def setBackground(self, filepath, dis=20, segments=32, rings=16):
        s = """
def setBackground(filepath, dis, segments, rings):
    bpy.ops.mesh.primitive_uv_sphere_add(size=dis, segments=segments, ring_count=rings)
    bpy.context.scene.objects['Sphere'].name = 'Background'
    img = bpy.data.images.load(filepath)
    cTex = bpy.data.textures.new('BackgroundColorTex', type = 'IMAGE')
    cTex.image = img
    mat = bpy.data.materials.new('BackgroundTexMat')
    mtex = mat.texture_slots.add()
    mtex.texture = cTex
    mtex.texture_coords = 'UV'
    mtex.use_map_ambient = True
    mat.emit = 1.0
    mat.specular_intensity = 0.0
    mat.diffuse_intensity = 0.0
    mtex.use_map_density = True
    mtex.mapping = 'FLAT'
    me = bpy.context.scene.objects['Background'].data
    me.materials.append(mat)
    for f in me.polygons: 
        f.material_index = len(me.materials)-1

    vertex_face_list = [None]*len(me.vertices)
    for i in range(len(vertex_face_list)):
        vertex_face_list[i] = []
    for i,f in enumerate(me.polygons):
        for j in f.vertices:
            vertex_face_list[j].append(i)
    start_vert = None
    for vert in range(len(me.vertices)):
        if start_vert is None or me.vertices[vert].co[2] > me.vertices[start_vert].co[2]:
            start_vert = vert
    
    def find_faces_with_verts(verts):
        if len(verts) == 0:
            return []
        ret = vertex_face_list[verts[0]]
        for vert in verts:
            ret = [x for x in ret if x in vertex_face_list[vert]]
        return ret
        
        
    bm = bmesh.new()
    bm.from_mesh(bpy.context.scene.objects['Background'].data)
    uv_layer = bm.loops.layers.uv.verify()
        
    def setUV(face, seg, ring, bm_):
        xs = 1.0*seg/segments
        xe = 1.0*(seg+1)/segments
        ys = 1.0*ring/rings
        ye = 1.0*(ring+1)/rings
        ys, ye = 1.0-ys, 1.0-ye
        verts = list(me.polygons[face].vertices)
        avgz = 0
        for v in [me.vertices[i].co[2] for i in verts]:
            avgz += v
        avgz /= len(verts)
        upper = [i for i,v in enumerate(verts) if me.vertices[v].co[2] > avgz]
        lower = [i for i,v in enumerate(verts) if me.vertices[v].co[2] < avgz]
        
        def setU(vts, y, bm_):
            if len(vts) == 1:
                bm_.faces[face].loops[vts[0]][uv_layer].uv = mathutils.Vector(((xs + xe)/2.0, y))
            else:
                if me.vertices[verts[vts[1]]].co[0]*me.vertices[verts[vts[0]]].co[1] - me.vertices[verts[vts[1]]].co[1]*me.vertices[verts[vts[0]]].co[0] < 0:
                    vts[0], vts[1] = vts[1], vts[0]
                bm_.faces[face].loops[vts[0]][uv_layer].uv = mathutils.Vector((xs, y))
                bm_.faces[face].loops[vts[1]][uv_layer].uv = mathutils.Vector((xe, y))
        setU(upper, ys, bm_)
        setU(lower, ye, bm_)
    
    processed_faces = []
    cur_face = vertex_face_list[start_vert][0]
    for seg in range(segments):
        setUV(cur_face, seg, 0, bm)
        processed_faces.append(cur_face)
        verts = list(me.polygons[cur_face].vertices)
        next_vert = verts[(verts.index(start_vert)+1)%len(verts)]
        verts.remove(start_vert)
        for ring in range(1, rings):
            cur_face = [x for x in find_faces_with_verts(verts) if x not in processed_faces][0]
            setUV(cur_face, seg, ring, bm)
            processed_faces.append(cur_face)
            verts = [v for v in list(me.polygons[cur_face].vertices) if v not in verts]
        if seg+1 < segments:
            cur_face = [x for x in find_faces_with_verts([start_vert, next_vert]) if x not in processed_faces][0]
    bm.to_mesh(bpy.context.scene.objects['Background'].data)
"""

        s += "\nsetBackground('%s', %f, %d, %d)" % (filepath, dis, segments, rings)
        self.__exec_cmd__(s, False)
        
    def smoothObject(self, obj):
        s = """
def smoothObject(obj):
    for p in bpy.context.scene.objects[obj].data.polygons:
        p.use_smooth = True
"""
        s += "\nsmoothObject('%s')" % obj
        self.__exec_cmd__(s, False)
        
    def subsurfObject(self, obj, levels=2):
        s = """
def subsurfObject(obj, levels):
    bpy.context.scene.objects.active = bpy.context.scene.objects[obj]
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.modifier_apply(apply_as='DATA')
    bpy.context.scene.objects[obj].modifiers['Subsurf'].levels = levels
    bpy.context.scene.objects[obj].modifiers['Subsurf'].render_levels = levels
"""
        s += "\nsubsurfObject('%s', %d)" % (obj, levels)
        self.__exec_cmd__(s, False)
        

    ####################### Commands for handling animation ############################################
    def setNumberFrames(self, nFrames):
        self.__exec_cmd__("bpy.context.scene.frame_start = 0", False)
        self.__exec_cmd__("bpy.context.scene.frame_end = %d" % (nFrames-1), False)
    
    def getNumberFrames(self):
        return eval(self.__exec_cmd__("bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1", True))

    def getCurrentFrame(self):
        return eval(self.__exec_cmd__("bpy.context.scene.frame_current", True))

    def setCurrentFrame(self, ff):
        self.__exec_cmd__("bpy.context.scene.frame_set(%d)" % ff, False)
        
    def setKeyframePose(self, obj, frame, loc=None, rot=None, interp='LINEAR'):
        self.__exec_cmd__("bpy.context.user_preferences.edit.keyframe_new_interpolation_type ='%s'" % interp, False)
        ret = True
        if loc is not None:
            self.setLocation(obj, loc)
            ret &= eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].keyframe_insert(data_path='location',frame=%d)" % (obj, frame), True))
        if rot is not None:
            self.setRotation(obj, rot)
            ret &= eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].keyframe_insert(data_path='rotation_quaternion',frame=%d)" % (obj, frame), True))
            
    def getKeyframeData(self, obj):
        s = """
def getKeyframeData(obj):
    if bpy.context.scene.objects[obj].animation_data is None:
        return []
    act = bpy.context.scene.objects[obj].animation_data.action
    rr = dict()
    for fcurve in act.fcurves:
        i = fcurve.array_index
        dp = fcurve.data_path
        for keyframe_point in fcurve.keyframe_points:
            frame, val = keyframe_point.co
            interp = keyframe_point.interpolation
            if frame not in rr:
                rr[frame] = dict()
            if dp not in rr[frame]:
                rr[frame][dp] = []
            while len(rr[frame][dp]) < i+1:
                rr[frame][dp].append(None)
            rr[frame][dp][i] = (val, interp)
    return [(frame, rr[frame]) for frame in sorted(rr.keys())]
"""
        self.__exec_cmd__(s, False)
        return eval(self.__exec_cmd__("getKeyframeData('%s')" % obj, True))
        
    def setKeyframesFromData(self, obj, data):
        for frame, keys in data:
            for data_path, vals in keys.items():
                v = [vals[i][0] if vals[i] is not None else eval(self.__exec_cmd__("bpy.context.scene.objects['%s'].%s[%d]" % (obj, data_path, i), True)) for i in range(len(vals))]
                interp = [x[1] for x in vals if x is not None][0]
                self.__exec_cmd__("bpy.context.user_preferences.edit.keyframe_new_interpolation_type ='%s'" % interp, False)
                self.__exec_cmd__("bpy.context.scene.objects['%s'].%s = %s" % (obj, data_path, str(tuple(v))), False)
                self.__exec_cmd__("bpy.context.scene.objects['%s'].keyframe_insert(data_path='%s',frame=%d)" % (obj, data_path, frame), False)
            
    def renderStillImage(self, filepath):
        self.__exec_cmd__("bpy.context.scene.render.filepath = '%s'" % filepath, False)
        self.__exec_cmd__("bpy.ops.render.render(write_still=True)", False)
    
    def renderAnimation(self, filepath, fps=30, save_images=True, multithread=True, make_video=True):
        if save_images:
            dirpath = os.path.dirname(filepath)
            ext = os.path.splitext(filepath)[0]
        else:
            dirpath = tempfile.mkdtemp()
            ext = os.path.join(dirpath, "tmp")
        self.__exec_cmd__("bpy.context.scene.render.fps = %f" % fps, False)
        self.__exec_cmd__("bpy.context.scene.render.filepath = '%s'" % ext, False)
        
        if multithread:
            self.__multiThreadRender(ext)
        else:
            self.__exec_cmd__("bpy.ops.render.render(animation=True)", False)
        
        global cv2_imported
        if cv2_imported and make_video:
            frames = sorted(glob.glob(ext + "*.png"))
            img = cv2.imread(frames[0])
            height, width = img.shape[:2]
            fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            for frame in frames:
                img = cv2.imread(frame)
                out.write(img)
            out.release()
        if not save_images:
            shutil.rmtree(dirpath)
            
    def __multiThreadRender(self, prefix, nthreads=min(multiprocessing.cpu_count(), 16)):
    
        cached_threads_mode = self.__exec_cmd__("bpy.context.scene.render.threads_mode", True)
        cached_nthreads = eval(self.__exec_cmd__("bpy.context.scene.render.threads", True))
        self.__exec_cmd__("bpy.context.scene.render.threads_mode = 'FIXED'", False)
        self.__exec_cmd__("bpy.context.scene.render.threads = 1", False)
    
        temp = tempfile.NamedTemporaryFile()
        self.saveScene(temp.name)
        nframes = eval(self.__exec_cmd__("bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1", True))
        q = Queue.Queue()
        for i in range(nframes):
            q.put(i)
        ndigits = len(str(nframes)) + 1
        finished = [False]*nframes
        
        paused = False
        
        def runThread(qq, ff):
            ii = None
            while True:
                while paused:
                    time.sleep(1.0)
                try:
                    ii = qq.get_nowait()
                except:
                    break
                try:
                    subprocess.call([self.blender_path, temp.name, '--background', '--render-output', prefix + '#'*ndigits + ".png", '--threads', '1', '--render-frame', str(ii)], stdout=DEVNULL, stdin=DEVNULL)
                    ff[ii] = True
                except:
                    qq.put(ii)
                    time.sleep(5.0)
                    
             
        def prog():
            return 1.0*len([f for f in finished if f])/nframes
        mon = cutil.ProgressMonitor(prog)
            
        threads = [threading.Thread(target=runThread, args=(q,finished)) for i in range(nthreads)]
        for i,thread in enumerate(threads):
            thread.daemon = True
            thread.start()
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_address = ('localhost', 1337)
            sock.bind(server_address)
            sock.setblocking(0)
            
            while not reduce(lambda x,y : x and y, finished):
                try:
                    data, address = sock.recvfrom(4096)
                    #print data
                    if data.lower() == 'pause':
                        paused = True
                        mon.pause()
                        print("\nPAUSING RENDER")
                    elif data.lower() == 'resume':
                        paused = False
                        mon.resume()
                        print("\nRESUMING RENDER")
                except socket.error:
                    pass
                
                time.sleep(0.1)
        finally:
            sock.close()
            
        for thread in threads:
            thread.join()
        
        temp.close()
        self.__exec_cmd__("bpy.context.scene.render.threads_mode = '%s'" % cached_threads_mode, False)
        self.__exec_cmd__("bpy.context.scene.render.threads = %d" % cached_nthreads, False)
        
    
    ######################### Commands for handling modifiers #########################################
    def addModifier(self, obj, name, mod_type):
        self.__exec_cmd__("bpy.context.scene.objects['%s'].modifiers.new(name='%s', type='%s')" % (obj, name, mod_type), False)
        
    def getModifiers(self, obj):
        return eval(self.__exec_cmd__("[(m.name, m.type) for m in bpy.context.scene.objects['%s'].modifiers]" % obj, True))
        
    def getModifierType(self, obj, mod):
        return self.__exec_cmd__("bpy.context.scene.objects['%s'].modifiers['%s'].type" % (obj, mod), True)
        
    def getModifierValue(self, obj, mod, val):
        return self.__exec_cmd__("bpy.context.scene.objects['%s'].modifiers['%s'].%s" % (obj, mod, val), True)
        
    def setModifierValue(self, obj, mod, val, new_val):
        return self.__exec_cmd__("bpy.context.scene.objects['%s'].modifiers['%s'].%s = %s" % (obj, mod, val, new_val), False)
        
        
    ######################## Commands for setting up fluid simulation ###################################3
    def getFluidMod(self, obj):
        mods = self.getModifiers(obj)
        if len(mods) == 0:
            return None
        if 'FLUID_SIMULATION' in zip(*mods)[1]:
            return mods[zip(*mods)[1].index('FLUID_SIMULATION')][0]
        else:
            return None
        
    def addAsFluidObj(self, obj):
        if self.getFluidMod(obj) is None:
            self.addModifier(obj, "fluid_sim", "FLUID_SIMULATION")
            
    # Fluid types
    FLUID_DOMAIN = 'DOMAIN'
    FLUID_NONE = 'NONE'
    FLUID_FLUID = 'FLUID'
    FLUID_INFLOW = 'INFLOW'
    FLUID_OUTFLOW = 'OUTFLOW'
    FLUID_PARTICLE = 'PARTICLE'
    FLUID_CONTROL = 'CONTROL'
    FLUID_OBSTACLE = 'OBSTACLE'
    
    def getFluidModType(self, obj):
        return self.getModifierValue(obj, self.getFluidMod(obj), "settings.type")
        
    def setFluidModType(self, obj, mod_type):
        return self.setModifierValue(obj, self.getFluidMod(obj), "settings.type", "'" + mod_type + "'")
    
    def getObjsOfFluidType(self, fluid_type):
        objs = self.getObjects()
        ret = []
        for obj in objs:
            fm = self.getFluidMod(obj)
            if fm is not None and self.getFluidModType(obj) == fluid_type:
                ret.append(obj)
        return ret
        
    def setFluidSetting(self, obj, setting, val, required_fluid_type):
        mod = self.getFluidMod(obj)
        if mod is None:
            raise ValueError("%s has not beend added as a fluid object." % obj)
        fluid_type = self.getFluidModType(obj)
        if fluid_type != required_fluid_type:
            raise ValueError("%s has fluid type %s, but must have type %s to set %s." % (obj, fluid_type, required_fluid_type, setting))
        self.setModifierValue(obj, mod, "settings." + setting, val)
        
    def setInflowVelocity(self, obj, vel):
        assert(len(vel) == 3)
        self.setFluidSetting(obj, "inflow_velocity", "mathutils.Vector(%s)" % str(tuple(vel)), self.FLUID_INFLOW)
        
    def setVolumeInitialization(self, obj, volume, shell):
        mod = self.getFluidMod(obj)
        if volume and shell:
            self.setModifierValue(obj, mod, "settings.volume_initialization", "'BOTH'")
        elif shell:
            self.setModifierValue(obj, mod, "settings.volume_initialization", "'SHELL'")
        else:
            self.setModifierValue(obj, mod, "settings.volume_initialization", "'VOLUME'")
            
    def setFluidResolution(self, res):
        self.setFluidSetting(self.getObjsOfFluidType(self.FLUID_DOMAIN)[0], "resolution", "%f" % res, self.FLUID_DOMAIN)
        
    def setFluidCachePath(self, filepath):
        self.setFluidSetting(self.getObjsOfFluidType(self.FLUID_DOMAIN)[0], "filepath", "'%s'" % filepath, self.FLUID_DOMAIN)
        
    def bakeFluidAnimations(self, fps=30, filepath=None):
        # First let's save and re-load the scene to clean it up.
        self.reloadScene()
        domain = self.getObjsOfFluidType(self.FLUID_DOMAIN)
        assert(len(domain) == 1)
        domain = domain[0]
        self.__exec_cmd__("bpy.context.scene.objects.active = bpy.context.scene.objects['%s']" % domain, False)
        nframes = eval(self.__exec_cmd__("bpy.context.scene.frame_end", True))
        self.setFluidSetting(domain, "end_time", "%f" % (1.0*nframes/fps), self.FLUID_DOMAIN)
        if filepath is not None:
            self.setFluidCachePath(filepath)
        else:
            filepath = self.getModifierValue(domain, self.getFluidMod(domain), "settings.filepath")
        self.bake_start = time.time()
        def keep_alive():
            start = self.bake_start
            while start is not None:
                dur = time.time() - start
                m, s = divmod(dur, 60)
                h, m = divmod(m, 60)
                cache_files = os.listdir(filepath)
                if len(cache_files) > 0:
                    percent = 100.0*max([int(''.join([c for c in f if c in '1234567890'])) for f in cache_files])/nframes
                else:
                    percent = 0.0
                if percent > 0.0:
                    dur_ = dur/(percent/100.0) - dur
                else:
                    dur_ = 0.0         
                m_, s_ = divmod(dur_, 60)
                h_, m_ = divmod(m_, 60)
                print("Elapsed time=%d:%02d:%02d, Percent=%.2f%%, Estimated Time Remaining=%d:%02d:%02d" % (h, m, s, percent, h_, m_, s_))
                time.sleep(60.0)
                start = self.bake_start
        thread = threading.Thread(target=keep_alive, args=())
        thread.daemon = True
        thread.start()
        self.__exec_cmd__("bpy.ops.fluid.bake()", False)
        self.bake_start = None
        #self.__exec_cmd__("bpy.ops.fluid.bake({'scene': bpy.data.scenes[0], 'active_object': bpy.context.scene.objects['%s'], 'blend_data' : bpy.data})" % domain, False)
            
            



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true", dest="show", default=False)
    parser.add_argument('--run_sim', action="store_true", dest="run_sim", default=True)
    parser.add_argument('--no-run_sim', action="store_false", dest="run_sim")
    parser.add_argument('--load', action="store", dest="load", default=None, type=str)
    parser.add_argument('--blender_path', action="store", dest="blender_path", default="blender", type=str)
    args = parser.parse_args()

    s = FluidSim(headless=(not args.show), run_sim=args.run_sim, filepath=args.load, blender_path=args.blender_path)
    #s.setNumberFrames(300)
#    s.setKeyframePose("Cube", 0, loc=(0,0,0))
#    s.setKeyframePose("Cube", 299, loc=(10,0,0))
    #s.addPyramid("Pyramid", (0,-2,0), (2,2,1))
    #s.setKeyframePose("Pyramid", 0, rot=(0,0,0))
    #s.setKeyframePose("Pyramid", 299, rot=(0, math.pi, 0))
    
#    s.addCube("Source", (0,0,3), (1,1,3))
#    
#    s.addAsFluidObj("Cube")
#    s.setObjectSize("Cube", (10,10,10))
#    s.setFluidModType("Cube", s.FLUID_DOMAIN)
#    
#    s.addAsFluidObj("Pyramid")
#    s.setFluidModType("Pyramid", s.FLUID_OBSTACLE)
#    
#    s.addAsFluidObj("Source")
#    s.setFluidModType("Source", s.FLUID_FLUID)
    #s.setInflowVelocity("Source", (0.0, 0.0, -3.0))
    
#    s.renderAnimation("/home/robolab/temp/test.avi")

#    s.__exec_cmd__("bpy.context.scene.objects['Cube'].dimensions = mathutils.Vector((10,10,10))", False)
#    #s.hideObject("Cube")
#    s.addCube("source", (0,0,0), (1,1,1))
#    
#    s.__exec_cmd__("bpy.context.scene.objects['Cube'].modifiers.new(name='f1', type='FLUID_SIMULATION')", False)
#    s.__exec_cmd__("bpy.context.scene.objects['Cube'].modifiers['f1'].settings.type = 'DOMAIN'", False)
#    
#    s.__exec_cmd__("bpy.context.scene.objects['Pyramid'].modifiers.new(name='f1', type='FLUID_SIMULATION')", False)
#    s.__exec_cmd__("bpy.context.scene.objects['Pyramid'].modifiers['f1'].settings.type = 'OBSTACLE'", False)
#    
#    s.__exec_cmd__("bpy.context.scene.objects['source'].modifiers.new(name='f1', type='FLUID_SIMULATION')", False)
#    s.__exec_cmd__("bpy.context.scene.objects['source'].modifiers['f1'].settings.type = 'INFLOW'", False)
#    s.__exec_cmd__("bpy.context.scene.objects['source'].modifiers['f1'].settings.inflow_velocity = mathutils.Vector((1,0,0))", False)
#    
#    s.__exec_cmd__("bpy.context.scene.objects['Cube'].hide_render = False", False)
    
    #s.__exec_cmd__("bpy.ops.fluid.bake({'scene': bpy.data.scenes[0], 'active_object': bpy.context.scene.objects['Cube'], 'blend_data' : bpy.data})", False)

    #s.setImageTexture("Cube", "/home/robolab/Downloads/hearthstone-uther-lightbringer-wallpaper.jpg")
    #s.setImageTexture("Cube", "/home/robolab/Downloads/test_pattern.png")
    #s.setTextureToAllFaces("Cube")
    #s.setBackground("/home/robolab/Downloads/hearthstone-uther-lightbringer-wallpaper.jpg", dis=3)
#    s.setBackground(os.path.join(os.path.dirname(os.path.realpath(__file__)), "textures", "lab.jpg"))
#    s.addPointLamp((-2,-4,5))
#    s.addPointLamp((-2,0,5))
#    s.addPointLamp((2,-4,5))
#    s.addPointLamp((2,0,5))
#    s.pointCameraAt(s.getLocation("Pyramid"), 10, math.pi/4, math.pi)
    #s.setImageTexture("Background", "/home/robolab/Downloads/hearthstone-uther-lightbringer-wallpaper.jpg")
    #s.setTextureToAllFaces("Background")
    #s.renderStillImage("/home/robolab/temp/test.png")
    cutil.keyboard()
    #s.saveScene('/home/robolab/temp/test.blend')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()

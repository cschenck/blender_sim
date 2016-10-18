#!/usr/bin/env python

# System imports
import math
import numpy as np
import os
import shutil
import tempfile
import time

# Third-party imports
import cutil
import imageio

# Local imports
from sim_lib import FluidSim

class TableEnvironment:
    TABLE_DIMS = (5,10,0.25)
    DOMAIN_HEIGHT = 5.0
    FLUID_RESOLUTION = 180
    FPS = 30
    BACKGROUND_DIS = 20.0
    VIDEO_PLANE_DIS = 20.0
    VIDEO_WIDTH = BACKGROUND_DIS*VIDEO_PLANE_DIS*(20.0/20.0/20.0)
    RENDER_RES = (640,480)
    
    OBJECTS = {"cup" :
                    {"height" : 1.3,
                     "top_radius" : 0.5,
                     "bottom_radius" : 0.4,
                     "thickness" : 0.1,
                     "segments" : 32,
                     "diffuse_color" : (0.5,0.0,0.0),
                     "diffuse_intensity" : 0.5,},
               "bottle" :
                    {"body_height" : 1.5,
                     "top_height" : 0.2,
                     "body_radius" : 0.4,
                     "top_radius" : 0.25,
                     "thickness" : 0.1,
                     "segments" : 32,
                     "diffuse_color" : (0.0,0.0,0.5),
                     "diffuse_intensity" : 0.5,},
                "mug" :
                    {"height" : 0.7,
                     "radius" : 0.5,
                     "thickness" : 0.1,
                     "segments" : 32,
                     "diffuse_color" : (0.0,0.5,0.0),
                     "diffuse_intensity" : 0.5,
                     "handle_arc_radius" : 0.25,
                     "handle_thickness" : 0.1,},
               }
               
    OBJECT_MODELS = {"bowl" : os.path.join("bowl", "Bowl.3DS"),
                     "dish" : os.path.join("dog_dish", "dog_dish.3ds"),
                     "fruit" : os.path.join("fruit_bowl", "fruit_bowl.3ds")}
               
    CUP_MAT_PRESETS = [
        {'image':'metal.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'ornate.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'paper.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'plastic.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'tiles.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'wood.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511},
        {'image':'cyclones_cup.png', 'mapping':'TUBE', 'texture_coords':'ORCO', 'specular_intensity':0.0, 'specular_hardness':511}]
    BOWL_MAT_PRESETS = [
        {'image':'metal.jpg', 'mapping':'CUBE', 'texture_coords':'ORCO'},
        {'image':'ornate.jpg', 'mapping':'SPHERE', 'texture_coords':'ORCO'},
        {'image':'paper.jpg', 'mapping':'CUBE', 'texture_coords':'ORCO'},
        {'image':'plastic.jpg', 'mapping':'CUBE', 'texture_coords':'ORCO'},
        {'image':'tiles.jpg', 'mapping':'SPHERE', 'texture_coords':'ORCO'},
        {'image':'wood.jpg', 'mapping':'TUBE', 'texture_coords':'ORCO'},
        {'image':'blue_bowl.png', 'mapping':'SPHERE', 'texture_coords':'ORCO'}]
    
    
    def __init__(self, blender_path="blender", filepath=None, pouring_object="cup", bowl_object="bowl", use_bowl=True, domain_bb=None):
        if filepath is not None:
            print("Attempting to load scene from " + filepath)
            self.sim = FluidSim(True, True, filepath, blender_path)
        else:
            print("Setting up the table environment...")
            # Set up the Simulator
            self.sim = FluidSim(True, True, None, blender_path) 
            
            # Add the background
            self.sim.setBackground(os.path.join(os.path.dirname(os.path.realpath(__file__)), "textures", "lab.jpg"), dis=self.BACKGROUND_DIS)
            
            # Add the table
            self.sim.addCube("Table", (0,0,0), TableEnvironment.TABLE_DIMS)
            
            if use_bowl:
                # Add the bowl
                self.sim.import3dsModel("Bowl", os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", self.OBJECT_MODELS[bowl_object]))
                self.sim.setLocation("Bowl", (0,0,TableEnvironment.TABLE_DIMS[-1]/2.0))
                self.sim.addAsFluidObj("Bowl")
                self.sim.setFluidModType("Bowl", FluidSim.FLUID_OBSTACLE)
                self.sim.setVolumeInitialization("Bowl", True, True)
                self.sim.smoothObject('Bowl')
                #self.sim.subsurfObject('Bowl')
                min_, max_ = self.sim.computeBoundingBox("Bowl")
            else:
                min_, max_ = domain_bb
            
            bowl_height = max_[-1]
            # Add the fluid domain
            max_ = tuple(list(max_[:-1]) + [min_[-1] + TableEnvironment.DOMAIN_HEIGHT])
            loc = [(min_[i] + max_[i])/2.0 for i in range(len(min_))]
            size = [max_[i] - min_[i] for i in range(len(min_))]
            size[0] = 4.0
            size[1] = 4.0
            self.sim.addCube("FluidDomain", loc, size)
            self.sim.addAsFluidObj("FluidDomain")
            self.sim.setFluidModType("FluidDomain", FluidSim.FLUID_DOMAIN)
            
            # Add the cup
            if pouring_object == "cup":
                loc = (0.0, loc[1]+size[1]/4.0, bowl_height + self.OBJECTS[pouring_object]['height'])
                self.addCup("Cup", loc)
            elif pouring_object == "bottle":
                loc = (0.0, self.OBJECTS[pouring_object]['body_height']/2.0 + self.OBJECTS[pouring_object]['top_height']*2.0, bowl_height + self.OBJECTS[pouring_object]['body_height'])
                self.addBottle("Cup", loc)
            elif pouring_object == "mug":
                loc = (0.0, loc[1]+size[1]/4.0, bowl_height + self.OBJECTS[pouring_object]['height']*3.0)
                self.addMug("Cup", loc)
            else:
                raise Exception("You forgot to add %s to the list of pouring objects." % pouring_object)
            self.sim.addAsFluidObj("Cup")
            self.sim.setFluidModType("Cup", FluidSim.FLUID_OBSTACLE)
            self.sim.setVolumeInitialization("Cup", True, True)
            self.sim.smoothObject('Cup')
            self.rotateAboutLip(False)
            
            self.sim.setFluidResolution(TableEnvironment.FLUID_RESOLUTION)
        
        self.rotate_point = (0.0,0.0,0.0)
        self.init_cup_point = self.sim.getLocation("Cup")
        self.sim.setCameraResolution(self.RENDER_RES)
        
        print("Done.")
        
    def prepareRender(self, replace_cup=None, background_video=None, cup_mat_preset=None, bowl_mat_preset=None, camera_angle=0.0, camera_pitch=math.pi/4.0, camera_dis=10.0, water_alpha=0.0, water_reflect=0.2, water_ior=1.33):
        self.sim.setCameraResolution(self.RENDER_RES)
        self.setLighting()
        
        self.sim.removeMaterial("Table")
        self.sim.setImageTexture("Table", os.path.join(os.path.dirname(os.path.realpath(__file__)), "textures", "table.jpg"), specular_intensity=0.0, specular_hardness=511)
        self.sim.setTextureToAllFaces("Table")
        self.sim.smoothObject('Table')
        
        self.resetFluidMaterial(water_alpha=water_alpha, water_reflect=water_reflect, water_ior=water_ior)
        
        self.sim.pointCameraAt((0,0,2), camera_dis, camera_pitch, camera_angle)
        
        if replace_cup is not None:
            self.replaceCupForRender(**replace_cup)
            
        if background_video is not None:
            self.addBackgroundVideo(**background_video)
            
        if cup_mat_preset is not None:
            self.__set_mat_from_preset("Cup", self.CUP_MAT_PRESETS[cup_mat_preset])
        if bowl_mat_preset is not None:
            self.__set_mat_from_preset("Bowl", self.BOWL_MAT_PRESETS[bowl_mat_preset])
            
    def __set_mat_from_preset(self, obj, mat):
        self.sim.removeMaterial(obj)
        fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "textures", mat['image'])
        self.sim.setImageTexture(obj, fp, **{k:v for k,v in mat.items() if k != 'image'})
        
    def setLighting(self):
        for obj in self.sim.getObjects(obj_type='LAMP'):
            self.sim.unlinkObject(obj)
        self.sim.addPointLamp((-TableEnvironment.TABLE_DIMS[0]/2.0, -TableEnvironment.TABLE_DIMS[1]/2.0, 10.0), energy=2.0)
        self.sim.addPointLamp((-TableEnvironment.TABLE_DIMS[0]/2.0, TableEnvironment.TABLE_DIMS[1]/2.0, 10.0), energy=2.0)
        self.sim.addPointLamp((TableEnvironment.TABLE_DIMS[0]/2.0, TableEnvironment.TABLE_DIMS[1]/2.0, 10.0), energy=2.0)
        self.sim.addPointLamp((TableEnvironment.TABLE_DIMS[0]/2.0, -TableEnvironment.TABLE_DIMS[1]/2.0, 10.0), energy=2.0)
        
        self.sim.addPointLamp((-TableEnvironment.TABLE_DIMS[0]/2.0, -TableEnvironment.TABLE_DIMS[1]/2.0, 1.5), energy=0.3)
        self.sim.addPointLamp((-TableEnvironment.TABLE_DIMS[0]/2.0, TableEnvironment.TABLE_DIMS[1]/2.0, 1.5), energy=0.3)
        self.sim.addPointLamp((TableEnvironment.TABLE_DIMS[0]/2.0, TableEnvironment.TABLE_DIMS[1]/2.0, 1.5), energy=0.3)
        self.sim.addPointLamp((TableEnvironment.TABLE_DIMS[0]/2.0, -TableEnvironment.TABLE_DIMS[1]/2.0, 1.5), energy=0.3)
        
    def __compute_cylinder_verts__(self, segments, top_radius, bottom_radius, thickness, height):
        ui = [None]*segments
        uo = [None]*segments
        li = [None]*segments
        lo = [None]*segments
        for i in range(segments):
            angle = 1.0*i/segments*2.0*math.pi
            ui[i] = (math.cos(angle)*(top_radius-thickness), math.sin(angle)*(top_radius-thickness), height/2.0)
            uo[i] = (math.cos(angle)*top_radius, math.sin(angle)*top_radius, height/2.0)
            li[i] = (math.cos(angle)*(bottom_radius-thickness), math.sin(angle)*(bottom_radius-thickness), -height/2.0)
            lo[i] = (math.cos(angle)*bottom_radius, math.sin(angle)*bottom_radius, -height/2.0)
        return (ui, uo, li, lo)
        
    def __compute_cylinder_faces__(self, upper, lower, point_inside=False):
        ret = []
        for i in range(len(upper)):
            j = (i + 1) % len(upper)
            face = [(id(upper), i), (id(lower), i), (id(lower), j), (id(upper), j)]
            if point_inside:
                face.reverse()
            ret.append(tuple(face))
        return ret
        
    def __compute_all_cylinder_faces(self, vert_rings):
        faces = []
        for i in range(len(vert_rings) - 1):
            v1 = vert_rings[i]
            v2 = vert_rings[i+1]
            faces += self.__compute_cylinder_faces__(v1, v2, point_inside=False)
        return faces
        
    def __add_cornering_verts(self, vert_rings, offset):
        vert_rings = np.array(vert_rings)        
        
        offset = np.array(offset)
        offset = np.reshape(np.repeat(offset, np.prod(vert_rings.shape[1:])), vert_rings.shape)
        
        first_center = np.mean(vert_rings[0,...], axis=0)
        last_center = np.mean(vert_rings[-1,...], axis=0)
        
        before_verts = np.zeros(vert_rings.shape)
        before_verts[0,...] = np.reshape(np.tile(first_center, vert_rings.shape[1]), vert_rings.shape[1:])
        before_verts[1:,...] = vert_rings[:-1,...]
        
        after_verts = np.zeros(vert_rings.shape)
        after_verts[:-1,...] = vert_rings[1:,...]
        after_verts[-1,...] = np.reshape(np.tile(last_center, vert_rings.shape[1]), vert_rings.shape[1:])
        
        before_verts = before_verts - vert_rings
        mags = np.reshape(np.repeat(np.sqrt(np.sum(before_verts**2, axis=-1)), before_verts.shape[-1]), before_verts.shape)
        before_verts = before_verts/mags*offset + vert_rings
        
        after_verts = after_verts - vert_rings
        mags = np.reshape(np.repeat(np.sqrt(np.sum(after_verts**2, axis=-1)), after_verts.shape[-1]), after_verts.shape)
        after_verts = after_verts/mags*offset + vert_rings
        
        # Interlace the output.
        ret = []
        for i in range(vert_rings.shape[0]):
            ret.append(before_verts[i,...].tolist())
            ret.append(vert_rings[i,...].tolist())
            ret.append(after_verts[i,...].tolist())
        return ret
        
    def __compute_cylinder_cap__(self, verts, point_down=False):
        ret = zip([id(verts)]*len(verts), list(range(len(verts))))
        if point_down:
            ret.reverse()
        return [tuple(ret)]
        
    def __compute_cylinder_cap_multiface__(self, verts, point_down=False, skip=1):
        center = [np.mean(np.array(verts), axis=0).tolist()]
        faces = []
        for i in range(0, len(verts), skip+1):
            idxs = np.array(range(i,i+skip+2)) % len(verts)
            f = [(id(center), 0)] + zip([id(verts)]*len(idxs), idxs.tolist())
            if point_down:
                f.reverse()
            faces.append(tuple(f))
        return faces, center
        
    def __convert_faces_verts__(self, faces, *verts):
        offset = [0]*len(verts)
        index = dict()
        for i in range(len(verts)):
            index[id(verts[i])] = i
            if i > 0:
                offset[i] = offset[i-1] + len(verts[i-1])
        
        ret_verts = []
        for v in verts:
            ret_verts += v
        
        ret_faces = []
        for j, face in enumerate(faces):
            loop = []
            for (l, i) in face:
                loop.append(offset[index[l]] + i)
            ret_faces.append(tuple(loop))
            
        return (ret_verts, ret_faces)
        
    def addCup(self, name, loc, glass=False, surface_subsample=False):
        p = self.OBJECTS["cup"]
        (ui, uo, li, lo) = self.__compute_cylinder_verts__(p["segments"], p["top_radius"], p["bottom_radius"], p["thickness"], p["height"])
        # Shift the z up by thickness for the bottom of the cup.
        for i in range(len(li)):
            li[i] = (li[i][0], li[i][1], li[i][2] + p["thickness"])
        
        vert_rings = [li, ui, uo, lo]
        if surface_subsample:
            vert_rings = self.__add_cornering_verts(vert_rings, [p["thickness"]*2.0, p["thickness"]*0.4, p["thickness"]*0.4, p["thickness"]*2.0])
            
        faces = []
        faces += self.__compute_all_cylinder_faces(vert_rings)
        
        if surface_subsample:
            f, v = self.__compute_cylinder_cap_multiface__(vert_rings[-1], point_down=True, skip=1)
            faces += f
            vert_rings.append(v)
            f, v = self.__compute_cylinder_cap_multiface__(vert_rings[0], point_down=False, skip=1)
            faces += f
            vert_rings.append(v)
        else:
            faces += self.__compute_cylinder_cap__(vert_rings[-1], point_down=True)
            faces += self.__compute_cylinder_cap__(vert_rings[0], point_down=False)
        
        verts, polys = self.__convert_faces_verts__(faces, *vert_rings)
        
        self.sim.addNewMeshObject(name, loc, verts, polys)
        if surface_subsample:
            self.sim.subsurfObject(name, levels=4)

        if glass:
            self.sim.setGlassMaterial(name)
        else:
            self.sim.setMaterial(name, diffuse_color=p['diffuse_color'], diffuse_intensity=p['diffuse_intensity'], specular_intensity=0.0, specular_hardness=511)
            
        if name == "Cup":
            self.fluid_shape = {'segments':p['segments'], 'height':p['height'], 'top_radius':(p['top_radius']-p['thickness']), 'bottom_radius':(p['bottom_radius']-p['thickness']), 'bottom_thickness':p['thickness'], 'lip_loc':(0.0,-p['top_radius']+p['thickness'], p['height']/2.0)}
            
    def addBottle(self, name, loc, glass=False, surface_subsample=False):
        p = self.OBJECTS["bottle"]
        
        # Body cylinder
        (in1, out1, in0, out0) = self.__compute_cylinder_verts__(p["segments"], p["body_radius"], p["body_radius"], p["thickness"], p["body_height"])
        # Shift the z up by thickness for the bottom of the bottle.
        for i in range(len(in0)):
            in0[i] = (in0[i][0], in0[i][1], in0[i][2] + p["thickness"])
            
        # Shoulder cylinder
        (in2, out2, _, _) = self.__compute_cylinder_verts__(p["segments"], p["top_radius"], p["body_radius"], p["thickness"], p["top_height"])
        # Move the z values up above the body
        for i in range(len(in1)):
            in2[i] = (in2[i][0], in2[i][1], p['body_height']/2.0 + p['top_height'])
            out2[i] = (out2[i][0], out2[i][1], p['body_height']/2.0 + p['top_height'])
            
        # Neck cylinder
        (in3, out3, _, _) = self.__compute_cylinder_verts__(p["segments"], p["top_radius"], p["top_radius"], p["thickness"], p["top_height"])
        # Move the z values up above the body
        for i in range(len(in1)):
            in3[i] = (in3[i][0], in3[i][1], p['body_height']/2.0 + 2.0*p['top_height'])
            out3[i] = (out3[i][0], out3[i][1], p['body_height']/2.0 + 2.0*p['top_height'])
        
        faces = []
        vert_rings = [in0, in1, in2, in3, out3, out2, out1, out0]
        if surface_subsample:
            vert_rings = self.__add_cornering_verts(vert_rings, [p["thickness"]*2.0, p["thickness"], p["thickness"], p["thickness"]*0.2, p["thickness"]*0.2, p["thickness"], p["thickness"], p["thickness"]*2.0])
        
        faces += self.__compute_all_cylinder_faces(vert_rings)
        
        if surface_subsample:
            f, v = self.__compute_cylinder_cap_multiface__(vert_rings[-1], point_down=True, skip=1)
            faces += f
            vert_rings.append(v)
            f, v = self.__compute_cylinder_cap_multiface__(vert_rings[0], point_down=False, skip=1)
            faces += f
            vert_rings.append(v)
        else:
            faces += self.__compute_cylinder_cap__(vert_rings[-1], point_down=True)
            faces += self.__compute_cylinder_cap__(vert_rings[0], point_down=False)
        
        verts, polys = self.__convert_faces_verts__(faces, *vert_rings)
        
        self.sim.addNewMeshObject(name, loc, verts, polys)
        if surface_subsample:
            self.sim.subsurfObject(name, levels=4)
            
        if glass:
            self.sim.setGlassMaterial(name)
        else:
            self.sim.setMaterial(name, diffuse_color=p['diffuse_color'], diffuse_intensity=p['diffuse_intensity'], specular_intensity=0.0, specular_hardness=511)

        if name == "Cup":
            self.fluid_shape = {'segments':p['segments'], 'height':p['body_height'], 'top_radius':(p['body_radius']-p['thickness']), 'bottom_radius':(p['body_radius']-p['thickness']), 'bottom_thickness':p['thickness'], 'lip_loc':(0.0,-p['top_radius']+p['thickness'], p['body_height']/2.0+2.0*p['top_height'])}
            
    def addMug(self, name, loc):
        p = self.OBJECTS["mug"]
        (ui, uo, li, lo) = self.__compute_cylinder_verts__(p["segments"], p["radius"], p["radius"], p["thickness"], p["height"])
        # Shift the z up by thickness for the bottom of the cup.
        for i in range(len(li)):
            li[i] = (li[i][0], li[i][1], li[i][2] + p["thickness"])
            
        handle_verts = []
        (_, handle_base, _, _) = self.__compute_cylinder_verts__(8, p["handle_thickness"]/2.0, p["handle_thickness"]/2.0, 0.0, 0.0)
        for i in range(p['segments']/2 + 1):
            angle = -math.pi/2.0 + 1.0*i/(p['segments']/2)*math.pi
            center = (math.cos(angle)*p['handle_arc_radius'] + p['radius'], 0.0, math.sin(angle)*p['handle_arc_radius'])
            vs = cutil.rotate_and_translate_point([cutil.listToPoint(pp) for pp in handle_base], cutil.listToPoint(center), cutil.eulerToQuaternion((0.0, -angle, 0.0)))
            vs = [cutil.pointToList(pp) for pp in vs]
            handle_verts.append(vs)
            
        
        faces = []
        faces += self.__compute_cylinder_faces__(ui, li, point_inside=True)
        faces += self.__compute_cylinder_faces__(uo, lo, point_inside=False)
        faces += self.__compute_cylinder_faces__(ui, uo, point_inside=False)
        faces += self.__compute_cylinder_cap__(li, point_down=False)
        faces += self.__compute_cylinder_cap__(lo, point_down=True)
        for i in range(1, len(handle_verts)):
            faces += self.__compute_cylinder_faces__(handle_verts[i], handle_verts[i-1], point_inside=False)
        faces += self.__compute_cylinder_cap__(handle_verts[-1], point_down=False)
        faces += self.__compute_cylinder_cap__(handle_verts[0], point_down=True)
        
        verts, polys = self.__convert_faces_verts__(faces, ui, uo, li, lo, *handle_verts)
        
        self.sim.addNewMeshObject(name, loc, verts, polys)
        self.sim.setMaterial(name, diffuse_color=p['diffuse_color'], diffuse_intensity=p['diffuse_intensity'], specular_intensity=0.0, specular_hardness=511)
        if name == "Cup":
            self.fluid_shape = {'segments':p['segments'], 'height':p['height'], 'top_radius':(p['radius']-p['thickness']), 'bottom_radius':(p['radius']-p['thickness']), 'bottom_thickness':p['thickness'], 'lip_loc':(0.0,-p['radius']+p['thickness'], p['height']/2.0)}
            
    def replaceCupForRender(self, cup_type, glass, surface_subsample):
        keyframe_data = self.sim.getKeyframeData("Cup")
    
        loc = self.sim.getLocation("Cup")
        rot = self.sim.getRotation("Cup")
        self.sim.unlinkObject("Cup")
        self.sim.reloadScene()
        if cup_type == "cup":
            self.addCup("Cup", (0,0,0), glass=glass, surface_subsample=surface_subsample)
        elif cup_type == "bottle":
            self.addBottle("Cup", (0,0,0), glass=glass, surface_subsample=surface_subsample)
        elif cup_type == "mug":
            self.addMug("Cup", (0,0,0))
        else:
            raise Exception("You forgot to add %s to the list of pouring objects." % cup_type)
            
        self.sim.setLocation("Cup", loc)
        self.sim.setRotation("Cup", rot)
        self.sim.setKeyframesFromData("Cup", keyframe_data)
        
    def resetFluidMaterial(self, water_alpha=0.0, water_reflect=0.2, water_ior=1.33):
        self.sim.removeMaterial("FluidDomain")
        self.sim.setFluidMaterial("FluidDomain", water_alpha=water_alpha, water_reflect=water_reflect, water_ior=water_ior)
        
    def setDuration(self, dur):
        self.sim.setNumberFrames(int(round(dur*self.FPS)))
        
    def setCupVisible(self, visible=True):
        self.sim.hideObject("Cup", hide=(not visible))
        
    def setCupAngle(self, angle, time):
        frame = int(round(time*self.FPS))
        newrot = cutil.eulerToQuaternion((angle, 0, 0))
        loc = self.sim.getLocation("Cup")
        
        if self.fix_above_origin:
            oldrot = cutil.listToQuaternion(self.sim.getRotation("Cup"))
            lip_loc = cutil.rotateAboutPoint(cutil.listToPoint([self.fluid_shape['lip_loc'][i] + loc[i] for i in range(len(loc))]), cutil.listToPoint(loc), oldrot)
            loc = (loc[0]-lip_loc[0]-self.fluid_shape['lip_loc'][0], loc[1]-lip_loc[1]-self.fluid_shape['lip_loc'][1], loc[2])
        
        self.sim.setKeyframePose("Cup", frame, loc=loc, rot=cutil.quaternionToList(newrot))
        print("Setting keyframe at frame %d with loc=%s and rot=%s" % (frame, str(loc), str(newrot)))
        
    def rotateAboutLip(self, enabled=True):
        self.fix_above_origin = enabled
        
    def setCameraAngle(self, angle):
        self.sim.pointCameraAt((0,0,2), 10.0, math.pi/4, angle)
        
    def rotateBackground(self, angle):
        self.sim.setRotation("Background", cutil.quaternionToList(cutil.eulerToQuaternion((0.0,0.0,angle))))
        
    def fillCup(self, ratio):
        assert("Fluid" not in self.sim.getObjects())
        
        p = self.fluid_shape
        (_, ui, _, li) = self.__compute_cylinder_verts__(p["segments"], p["top_radius"], p["bottom_radius"], 0.0, p["height"])
        # Shift the z up by thickness for the bottom of the cup.
        for i in range(len(li)):
            li[i] = (li[i][0], li[i][1], li[i][2] + p["bottom_thickness"])
        h = p["height"] - p["bottom_thickness"]
        # Shift the z down by the ratio
        for i in range(len(ui)):
            ui[i] = (ui[i][0], ui[i][1], ui[i][2] - h*(1.0 - ratio))
        faces = []
        faces += self.__compute_cylinder_faces__(ui, li, point_inside=False)
        faces += self.__compute_cylinder_cap__(ui, point_down=False)
        faces += self.__compute_cylinder_cap__(li, point_down=True)
        verts, polys = self.__convert_faces_verts__(faces, ui, li)
        
        loc = self.sim.getLocation("Cup")
        self.sim.addNewMeshObject("Fluid", loc, verts, polys)
        self.sim.hideObject("Fluid", True)
        self.sim.addAsFluidObj("Fluid")
        self.sim.setFluidModType("Fluid", FluidSim.FLUID_FLUID)
        self.sim.setVolumeInitialization("Fluid", True, True)
        
    def addBackgroundVideo(self, filepath, start_ratio):
        self.sim.addNewMeshObject("VideoPlane", (0.0,0.0,0.0), [(-0.5,0.5,0.0), (0.5,0.5,0.0), (0.5,-0.5,0.0), (-0.5,-0.5,0.0)], [(3,0,1,2)])
        self.sim.setVideoTexture("VideoPlane", filepath, start_ratio)
        width, height = self.sim.getImageTextureSize("VideoPlane")
        
        video_width = self.VIDEO_PLANE_DIS*math.tan(self.sim.getCameraFOV()/2.0)*2.0
        video_height = 1.0*self.RENDER_RES[1]/self.RENDER_RES[0]*video_width
        
        self.sim.setObjectSize("VideoPlane", (video_width, video_height, 0.0))
        cam_rot = self.sim.getRotation("Camera")
        self.sim.setRotation("VideoPlane", cam_rot)
        cam_loc = self.sim.getLocation("Camera")
        loc = (cam_loc[0], cam_loc[1], cam_loc[2] - self.VIDEO_PLANE_DIS)
        loc = cutil.pointToList(cutil.rotateAboutPoint(cutil.listToPoint(loc), cutil.listToPoint(cam_loc), cutil.listToQuaternion(cam_rot)))
        self.sim.setLocation("VideoPlane", loc)    
        self.sim.setTextureToAllFaces("VideoPlane")
        self.sim.setMaterialEmit("VideoPlane", 1.0)   
        self.sim.setMaterialShadeless("VideoPlane") 
        
    def generateOutput(self, render_path, bake_path, render=True, bake=True, dry_run=False, data_prefix="data", gt_prefix="ground_truth", composite_video_plane=False, render_water=True):
        if bake:
            if not dry_run:
                self.sim.setFluidResolution(TableEnvironment.FLUID_RESOLUTION)
                print("Baking fluid simulation, this may take awhile...")
                self.sim.bakeFluidAnimations(filepath=bake_path)
                self.sim.smoothObject('FluidDomain')
                self.sim.subsurfObject('FluidDomain')
            else:
                fp = open(os.path.join(bake_path, "bake_file.txt"), "w")
                fp.write("test")
                fp.close()
        
        if render:
            if not dry_run:
                self.sim.setCameraResolution(self.RENDER_RES)
                self.sim.setFluidCachePath(filepath=bake_path)
                
                render_data = data_prefix is not None
                render_gt = gt_prefix is not None
                
                dirpath = tempfile.mkdtemp()
                self.sim.saveScene(os.path.join(dirpath, "scene.blend"))
                
                if not render_water:
                    self.sim.unlinkObject("FluidDomain")
                    self.sim.reloadScene()
                
                try:
                    
                    out = None
                    if render_data:
                        # First render full scene.
                        self.sim.renderAnimation(os.path.join(dirpath, "full"), save_images=True, make_video=False)
                        out = imageio.get_writer(os.path.join(render_path, "%s.avi" % data_prefix), fps=self.FPS)
                        if "VideoPlane" in self.sim.getObjects():
                            fp, offset = self.sim.getVideoTextureSettings("VideoPlane")
                            vp = imageio.get_reader(cutil.findFile(fp, os.path.dirname(os.path.realpath(__file__))))
                            i = 0
                            while i < offset:
                                vp.get_next_data()
                                i += 1
                        else:
                            vp = None
            
                    # Now render black/white scene for getting ground truth of water.
                    objs_to_render = [("Cup", 0), ("Bowl", 1), ("Table", None)]
                    if render_water:
                        objs_to_render.append(("FluidDomain", 2))
                    self.sim.enableLights(enable=False)
                    for obj in self.sim.getObjects():
                        self.sim.removeMaterial(obj)
                        self.sim.hideObject(obj, hide=True)
                        
                    self.sim.hideObject("Background", hide=False)
                    for obj, ch in objs_to_render:
                        if ch is None:
                            color = [1.0,1.0,1.0]
                        else:
                            color = [0.0,0.0,0.0]
                            color[ch] = 1.0
                        self.sim.setMaterial(obj, diffuse_color=color, diffuse_intensity=1.0)
                        self.sim.setMaterialEmit(obj, 1.0)
                        self.sim.hideObject(obj, hide=False)
                        
                        # Render only obj
                        self.sim.renderAnimation(os.path.join(dirpath, obj + "_only"), save_images=True, make_video=False)
                        self.sim.removeMaterial(obj)
                        self.sim.hideObject(obj, hide=True)

                    # Render one more time to figure out the overlap ordering.
                    for obj, ch in objs_to_render:
                        if ch is None:
                            continue
                        color = [0.0,0.0,0.0]
                        color[ch] = 1.0
                        self.sim.setMaterial(obj, diffuse_color=color, diffuse_intensity=1.0)
                        self.sim.setMaterialEmit(obj, 1.0)
                        self.sim.hideObject(obj, hide=False)
                    self.sim.renderAnimation(os.path.join(dirpath, "overlap_ordering"), save_images=True, make_video=False)
                    
                    
                    ii = None
                    # Now combine the color channels.
                    pub = cutil.SingleLineUpdater()
                    ret_all = {}
                    ret_visible = {}
                    for i in range(self.sim.getNumberFrames()):
                        pub.publish("Stitching images %d of %d." % (i, self.sim.getNumberFrames()))
                        gt = np.zeros((self.RENDER_RES[1], self.RENDER_RES[0], 4,), dtype=np.float32)
                        sim = np.zeros((self.RENDER_RES[1], self.RENDER_RES[0], 3,), dtype=np.float32)
                        for obj, ch in objs_to_render:
                            fp = cutil.find_file_by_index(os.path.join(dirpath, obj + "_only%0*d.png"), i)
                            img = imageio.imread(fp)
                            if ch is not None:
                                gt[...,ch] = img[...,ch]
                            sim += img[...,:3]

                        # Use the alpha channel to indicate overlap ordering.
                        fp = cutil.find_file_by_index(os.path.join(dirpath, "overlap_ordering%0*d.png"), i)
                        img = imageio.imread(fp)
                        for obj, ch in objs_to_render:
                            if ch is None:
                                continue
                            idxs = np.where(img[...,ch] > 128)
                            gt[idxs + (-1*np.ones((idxs[0].shape[0],), dtype=int),)] = 255.0*ch/2.0

                        if render_gt:
                            imageio.imwrite(os.path.join(render_path, "%s%04d.png" % (gt_prefix, i)), gt)
                        if render_data:
                            full_img = imageio.imread(cutil.find_file_by_index(os.path.join(dirpath, "full%0*d.png"), i))
                            if vp is not None and composite_video_plane: 
                                pxls = np.where(np.max(sim, axis=2) < 128)
                                try:
                                    ii = vp.get_next_data()
                                except:
                                    if ii is None:
                                        raise
                                full_img[pxls[0], pxls[1], :ii.shape[-1]] = ii[pxls[0], pxls[1], :]
                            imageio.imwrite(os.path.join(render_path, "%s%04d.png" % (data_prefix, i)), full_img)
                            out.append_data(full_img)

                        # Finally, compute the bounding boxes from the gt.
                        computeBoundingBoxesInImage(gt, i, ret_all, ret_visible)

                    # Write out bounding box files.
                    BB_FILENAME_ALL = 'bounding_boxes_all.txt'
                    BB_FILENAME_VISIBLE = 'bounding_boxes_visible.txt'
                    fp_all = open(os.path.join(render_path, BB_FILENAME_ALL), 'w')
                    fp_visible = open(os.path.join(render_path, BB_FILENAME_VISIBLE), 'w')
                    for ii in sorted(ret_all.keys()):
                        fp_all.write("(%d, %s)\n" % (ii, str(ret_all[ii])))
                        fp_visible.write("(%d, %s)\n" % (ii, str(ret_visible[ii])))
                    fp_all.close()
                    fp_visible.close()
                            
                    if vp is not None:
                        vp.close()
                    if out is not None:
                        out.close()
                finally:
                    # Reset scene
                    self.sim.loadScene(os.path.join(dirpath, "scene.blend"))
                    shutil.rmtree(dirpath)
                print("Finished rendering.")
            else:
                fp = open(os.path.join(render_path, "render_file.txt"), "w")
                fp.write("test")
                fp.close()

    def trackCupLip(self, render_path):
        # First get the vertex we want to track.
        self.sim.setCurrentFrame(0)
        verts = np.array(self.sim.getVertices("Cup"))
        max_z = verts[:,-1].max()
        min_z = verts[:,-1].min()
        # Get point closest to (0,0,max_z).
        idx = np.dot(np.square(verts - np.array([[0.0,0.0,max_z],]*verts.shape[0])), np.array([[1.0], [1.0], [10.0]])).argmin()

        fp = open(os.path.join(render_path, 'lip_loc.csv'), 'w')
        pub = cutil.SingleLineUpdater()
        for i in range(self.sim.getNumberFrames()):
            pub.publish("Computing lip at frame %d of %d." % (i, self.sim.getNumberFrames()))
            self.sim.setCurrentFrame(i)
            x3, y3, z3 = self.sim.getVertex("Cup", idx) #self.sim.pointInObjectFrame("Camera", self.sim.getVertex("Cup", idx))
            x2, y2 = self.sim.pointToPixel((x3, y3, z3))
            fp.write("%d, %d, %d\n" % (i, x2, y2))
        fp.flush()
        fp.close()
        print("Done computing lip.")

def computeBoundingBoxesInImage(img, ii, ret_all, ret_visible):
    CHANNEL_NAMES = ["r", "g", "b"]
    ret_all[ii] = {}
    ret_visible[ii] = {}
    for ch in range(3):
        idxs = np.where(img[...,ch] > 128)
        if idxs[0].shape[0] > 0:
            ret_all[ii][CHANNEL_NAMES[ch]] = ((idxs[0].min(), idxs[1].min()), (idxs[0].max(), idxs[1].max()))
        else:
            ret_all[ii][CHANNEL_NAMES[ch]] = ((-1, -1), (-1, -1))

    # Now restrict the image to just what is visible.
    for ch in range(3):
        idxs = np.where((img[...,-1] >= ch/3.0*255) & (img[...,-1] <= (ch+1)/3.0*255))
        for ch2 in range(3):
            if ch2 == ch:
                continue
            img[idxs + (ch2*np.ones((idxs[0].shape[0],), dtype=int),)] = 0

    for ch in range(3):
        idxs = np.where(img[...,ch] > 128)
        if idxs[0].shape[0] > 0:
            ret_visible[ii][CHANNEL_NAMES[ch]] = ((idxs[0].min(), idxs[1].min()), (idxs[0].max(), idxs[1].max()))
        else:
            ret_visible[ii][CHANNEL_NAMES[ch]] = ((-1, -1), (-1, -1))

            

            
class AlternateBowlTableEnvironment(TableEnvironment):
    DOMAIN_DIMS = ((-1.0, -1.0, 0.0), (1.0,3.0,TableEnvironment.DOMAIN_HEIGHT))
    def __init__(self, blender_path="blender", filepath=None, pouring_object="cup", alternate_bowl="cup"):
        TableEnvironment.__init__(self, blender_path=blender_path, filepath=filepath, pouring_object=pouring_object, use_bowl=False, domain_bb=self.DOMAIN_DIMS)
        if alternate_bowl == "cup":
            loc = (0.0, 0.0, self.OBJECTS[alternate_bowl]['height']/2.0 + TableEnvironment.TABLE_DIMS[-1]/2.0)
            self.addCup("Bowl", loc)
        elif alternate_bowl == "bottle":
            loc = (0.0, 0.0, self.OBJECTS[alternate_bowl]['body_height']/2.0 + TableEnvironment.TABLE_DIMS[-1]/2.0)
            self.addBottle("Bowl", loc)
        elif alternate_bowl == "mug":
            loc = (0.0, 0.0, self.OBJECTS[alternate_bowl]['height']/2.0 + TableEnvironment.TABLE_DIMS[-1]/2.0)
            self.addMug("Bowl", loc)
        else:
            raise Exception("You forgot to add %s to the list of alternate bowls." % alternate_bowl)
        self.sim.smoothObject('Bowl')
        self.sim.addAsFluidObj("Bowl")
        self.sim.setFluidModType("Bowl", FluidSim.FLUID_OBSTACLE)
        self.sim.setVolumeInitialization("Bowl", True, True)
        
        loc = (0.0, -self.fluid_shape['lip_loc'][1]*2.0, self.sim.getLocation("Cup")[-1])
        self.sim.setLocation("Cup", loc)
        self.rotateAboutLip(True)
        
def main():
    blend_file = "/media/robolab/arthus/fluid_sim_output/scene18/scene.blend"
    env = TableEnvironment(filepath=blend_file, blender_path=os.path.join(cutil.findFile("blender-2.69", os.path.join(os.path.expanduser("~"), "proj")), "blender"), pouring_object="bottle")
    #env = AlternateBowlTableEnvironment(blender_path=os.path.join(cutil.findFile("blender-2.69", os.path.expanduser("~")), "blender"), pouring_object="cup", alternate_bowl="cup")
    
    #env.sim.renderStillImage("/home/robolab/temp/test.png")
    
    #env.setCameraAngle(-3.0*math.pi/4.0)
    #env.addBackgroundVideo('/home/robolab/temp/oni2avi/test-img.avi', 0.0)

    #env.setDuration(1.0)
#    env.fillCup(0.6)
#    t = 0.0
#    while t <= 3.0:
#        env.setCupAngle(t/3.0*math.pi, t)
#        t += 0.1
    #env.setCupAngle(0.0, 0.0)
    #env.setCupAngle(math.pi/2.0, 3.0)
    #env.sim.hideObject("FluidDomain")
    env.replaceCupForRender("bottle", glass=True, surface_subsample=True)
    env.setLighting()
    #env.generateOutput(render_path="/home/robolab/temp/render", bake_path=os.path.join(os.path.dirname(blend_file), "bake_files"), render=True, bake=False, dry_run=False, data_prefix=None)
    
#    env.sim.addCube("Inflow", (0,0,TableEnvironment.DOMAIN_HEIGHT - 1), (1,1,1))
#    env.sim.addAsFluidObj("Inflow")
#    env.sim.setFluidModType("Inflow", FluidSim.FLUID_FLUID)
#    env.sim.setVolumeInitialization("Inflow", True, True)
#    env.sim.bakeFluidAnimations()
#    #cutil.keyboard()
    
    env.sim.saveScene('/home/robolab/temp/test.blend')
    

    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()

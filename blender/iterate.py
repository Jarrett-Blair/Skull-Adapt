import os
import bpy
import math
import pandas as pd


#Setup


#Read in rotations and get species names
rotations = pd.read_csv("C:\Carabid_Data\Skulls\Rotations.csv")
sdir = "C:\\Users\\blair\OneDrive - CMN\Scans\\"
species = os.listdir(sdir)

#Create rotation legend
rotate = [0, [0,0,0], [0,0,180], [180,0,180], [180,0,0]]


#Loop

species = species[11:]
#Loop through species folders
for sp in species:
    objs = []
    #Get obj filenames
    for file in os.listdir(sdir + sp + "\\OBJ"):
        if file.endswith(".obj"):
            objs.append(file)
    #Add leading zeros to short filenames        
    #strlen = len(max(objs, key = len))
    #for i in range(len(objs)):
        #if len(objs[i]) < strlen:
            #zlen = strlen - len(objs[i])
            #zeros = "0" * zlen
            #temp = objs[i][0:((strlen - 5) - (5-zlen))] + zeros
            #objs[i] = temp + objs[i][((strlen - 5) - (5-zlen)):strlen]
    #objs = sorted(objs)
    
    #Loop through objs in current species
    for i in range(len(objs)):
        rotdex = rotations[sp][i]
        if rotdex == 0:
            continue
        file = objs[i]
        name = file[:-4]
        #Read in obj file
        file_loc = sdir + sp + "\\OBJ\\" + file
        bpy.ops.import_scene.obj(filepath=file_loc)
        #Assign current object to 'obj'
        obj = bpy.context.scene.objects[name]
        #Get object length/polycount and set scale value
        length = obj.dimensions.y
        scale = 35/length
        poly = len(obj.data.polygons)
        #Decimate, rotate, and scale
        x,y,z = rotate[rotdex]
        obj.rotation_euler = [math.radians(x),math.radians(y),math.radians(z)]
        obj.modifiers.new(name = 'Decimate', type='DECIMATE')
        obj.modifiers["Decimate"].ratio = 17500/poly
        bpy.data.objects[name].scale = (scale, scale, scale)
        
        bpy.context.scene.render.filepath = r"C:\Carabid_Data\Skulls\Renders\\" + name + "-"
        bpy.ops.render.render(animation=True)
        bpy.data.objects.remove(obj, do_unlink = True)
    


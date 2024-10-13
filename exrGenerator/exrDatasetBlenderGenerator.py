
bl_info = {
    "name": "GPT EXR Generator" ,
    "blender": (3, 0, 0),
    "category": "Object",
}

import bpy
import os
import random
import math

class Solution:
   def __init__(self, radius: float, x_center: float, y_center: float):
        self.x_center = x_center
        self.y_center = y_center
        self.x_min = x_center - radius
        self.x_max = x_center + radius
        self.y_min = y_center - radius
        self.y_max = y_center + radius
        self.radius = radius
   def randPoint(self) -> list[float]:
        output = [0, 0]
        while True:
            output[0], output[1] = random.uniform(self.x_min, self.x_max), random.uniform(self.y_min, self.y_max)
            if math.sqrt(pow(output[0]-self.x_center,2) + pow(output[1]-self.y_center,2)) <= self.radius:
                return output
            
random_brush_lib = []

class GPTExrGeneratorProperties(bpy.types.PropertyGroup):
    random_brush_lib_path: bpy.props.StringProperty(
        name="Random Brush Lib Path",
        description="Path to the folder containing EXR files",
        default="",
        subtype='DIR_PATH'
    ) # type: ignore
    random_stroke_pressure: bpy.props.IntProperty(
        name="Random Stroke Pressure",
        description="Pressure for random strokes",
        default=1,
        min=0
    ) # type: ignore
    random_stroke_size: bpy.props.IntProperty(
        name="Random Stroke Size",
        description="Size for random strokes",
        default=1,
        min=0
    ) # type: ignore
    random_stroke_time: bpy.props.IntProperty(
        name="Random Stroke Time",
        description="Time for random strokes",
        default=1,
        min=0
    ) # type: ignore
    number_of_random_strokes: bpy.props.IntProperty(
        name="Number of Random Strokes",
        description="Number of random strokes",
        default=1,
        min=1
    ) # type: ignore
    number_of_steps: bpy.props.IntProperty(
        name="Number of Steps",
        description="Number of steps",
        default=1,
        min=1
    ) # type: ignore
    exr_data_folder: bpy.props.StringProperty(
        name="EXR Data Folder",
        description="Folder to save EXR data",
        default="",
        subtype='DIR_PATH'
    ) # type: ignore
    png_data_folder: bpy.props.StringProperty(
        name="PNG Data Folder",
        description="Folder containing PNG files",
        default="",
        subtype='DIR_PATH'
    ) # type: ignore
    image_height: bpy.props.IntProperty(
        name="Image Height",
        description="Height for EXR conversion",
        default=1024,
        min=1
    ) # type: ignore
    image_width: bpy.props.IntProperty(
        name="Image Width",
        description="Width for EXR conversion",
        default=1024,
        min=1
    ) # type: ignore

class GPTExrGeneratorPanel(bpy.types.Panel):
    bl_label = "GPT EXR Generator"
    bl_idname = "OBJECT_PT_gpt_exr_generator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GPT EXR'

    def draw(self, context):
        layout = self.layout
        props = context.scene.gpt_exr_generator
        
        layout.operator("object.create_exr_data", text="Create Brush In Lib Folder")
        layout.prop(props, "random_brush_lib_path")
        layout.operator("object.generate_random_brush_lib", text="Generate Random Brush Lib")
        layout.prop(props, "random_stroke_pressure")
        layout.prop(props, "random_stroke_size")
        layout.prop(props, "random_stroke_time")
        layout.prop(props, "number_of_random_strokes")
        layout.prop(props, "number_of_steps")
        layout.prop(props, "exr_data_folder")
        layout.operator("object.generate_exr_data", text="Generate")
        layout.prop(props, "png_data_folder")
        layout.prop(props, "image_height")
        layout.prop(props, "image_width")
        layout.operator("object.convert_png_to_exr", text="Convert PNG to EXR")

class CreateExrDataOperator(bpy.types.Operator):
    bl_idname = "object.create_exr_data"
    bl_label = "Create Brush In Lib Folder"
    

    
    def execute(self, context):
        props = context.scene.gpt_exr_generator
        #bpy.context.scene.render.filepath = props.random_brush_lib_path
        bpy.data.is_saved=True
        bpy.data.filepath = props.random_brush_lib_path
        # Create VDM brush (pseudo-code, actual implementation may vary)
        # create_vdm_brush(step, exr_data_folder)
        bpy.ops.vdmbrush.create()
        exr_path = props.random_brush_lib_path + bpy.context.scene.VDMBrushBakerAddonData.draft_brush_name +".exr"
        print("EXR_PATH "+ exr_path)
        bpy.ops.render.render(write_still=True)
        bpy.ops.image.save_as(save_as_render=False, filepath=exr_path , relative_path=False, show_multiview=False, use_multiview=False)
        #bpy.ops.image.save_as(save_as_render=False, copy=False, filepath=exr_path, check_existing=True, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='FILE_SORT_ALPHA')


class GenerateRandomBrushLibOperator(bpy.types.Operator):
    bl_idname = "object.generate_random_brush_lib"
    bl_label = "Generate Random Brush Lib"
    
    def execute(self, context):
        props = context.scene.gpt_exr_generator
        exr_path = props.random_brush_lib_path
        random_brush_lib = []

        if not os.path.exists(exr_path):
            self.report({'ERROR'}, "Invalid path")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='SCULPT')
        for file_name in os.listdir(exr_path):
            if file_name.endswith('.exr'):
                file_path = os.path.join(exr_path, file_name)
                brush_name = os.path.splitext(file_name)[0]

                # Create a new brush
                new_brush = bpy.data.brushes.new(name=brush_name)
                new_texture = bpy.data.textures.new(name=brush_name, type='IMAGE')
                new_texture.image = bpy.data.images.load(file_path)

                # Set brush properties
                new_brush.texture = new_texture
                new_brush.strength = 1.0
                new_brush.use_pressure_strength = False
                
                #bpy.data.brushes[brush_name].texture_slot.map_mode = 'AREA_PLANE'
                
                new_brush.texture_slot.map_mode = 'AREA_PLANE'
                #new_brush.texture_slot.map_mode = 'VIEW_PLANE'
                new_brush.use_color_as_displacement = True
                new_brush.stroke_method = 'ANCHORED'
                new_brush.curve_preset = 'CONSTANT'
                new_brush.use_color_as_displacement = True



                random_brush_lib.append(new_brush)
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene['random_brush_lib'] = random_brush_lib
        self.report({'INFO'}, f"Generated {len(random_brush_lib)} brushes")
        return {'FINISHED'}

            
class GenerateExrDataOperator(bpy.types.Operator):
    bl_idname = "object.generate_exr_data"
    bl_label = "Generate EXR Data"

    def execute(self, context):
        props = context.scene.gpt_exr_generator
        random_brush_lib = context.scene.get('random_brush_lib', [])
        exr_data_folder = props.exr_data_folder

        if not random_brush_lib:
            self.report({'ERROR'}, "No brushes available. Generate brushes first.")
            return {'CANCELLED'}

        if not os.path.exists(exr_data_folder):
            os.makedirs(exr_data_folder)
            
        obj = Solution(0.5, 0,0)
        
        for step in range(props.number_of_steps):
            # Remove existing Grid object if present
            if "Grid" in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)

            # Create a new Grid object
            #bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            #grid = bpy.context.active_object
           # grid.name = "Grid"
            bpy.ops.sculptplane.create()
            plane = bpy.context.active_object
            bpy.context.view_layer.objects.active = plane
            bpy.ops.object.mode_set(mode='SCULPT')
            for _ in range(props.number_of_random_strokes):
                # Select a random brush
                brush = random.choice(random_brush_lib)
                bpy.context.tool_settings.sculpt.brush = brush
                # Set stroke properties
                stroke_pressure = props.random_stroke_pressure
                stroke_size = props.random_stroke_size
                stroke_time = props.random_stroke_time

                param_1 = obj.randPoint()
                randomX = (param_1[0])
                randomY = (param_1[1])
                print("DUPA " +str(randomX) +" " +str(randomY))
                #randomX = random.uniform(-1, 1)
                #randomY = random.uniform(-1, 1)
                strokes=[]
                # Make stroke
                strokes.append({
                    "name": "stroke",
                    "location": (randomX, randomY, 0.0),
                    "mouse": (randomX, randomY),
                    "pressure": stroke_pressure,
                    "size": stroke_size,
                    "pen_flip": False,
                    "time": 0,
                    "is_start": True,
                    "mouse_event": (randomX, randomY),
                    "x_tilt": 0,
                    "y_tilt": 0
                }) 
                strokes.append({
                    "name": "stroke",
                    "location": (randomX, randomY, 0.1),
                    "mouse": (randomX, randomY),
                    "pressure": stroke_pressure,
                    "size": stroke_size,
                    "pen_flip": False,
                    "time": stroke_time,
                    "is_start": False,
                    "mouse_event": (randomX, randomY),
                    "x_tilt": 0,
                    "y_tilt": 0
                })
                bpy.ops.sculpt.brush_stroke(stroke=strokes)
                # Perform stroke (pseudo-code, actual implementation may vary)
                # perform_stroke(grid, brush, random_x, random_y, stroke_pressure, stroke_size, stroke_time)
            bpy.ops.object.mode_set(mode='OBJECT')
            # Capture images from all cameras
            for camera in bpy.data.cameras:
                # Set camera as active
                bpy.context.scene.camera = bpy.data.objects[camera.name]

                # Render and save image
                image_name = f"{step}_{camera.name}.png"
                image_path = os.path.join(exr_data_folder, image_name)
                bpy.context.scene.render.filepath = image_path
                bpy.ops.render.render(write_still=True)

            # Create VDM brush (pseudo-code, actual implementation may vary)
            # create_vdm_brush(step, exr_data_folder)
            #bpy.ops.vdmbrush.create()
            #exr_path = props.random_brush_lib_path
            #bpy.ops.image.save_as(save_as_render=False, filepath=exr_path +"/"+get_add.draft_brush_name+".exr", relative_path=False, show_multiview=False, use_multiview=False)
            
        self.report({'INFO'}, "EXR data generation complete.")
        return {'FINISHED'}

class ConvertPngToExrOperator(bpy.types.Operator):
    bl_idname = "object.convert_png_to_exr"
    bl_label = "Convert PNG to EXR"

    def execute(self, context):
        props = context.scene.gpt_exr_generator
        png_data_folder = props.png_data_folder
        image_height = props.image_height
        image_width = props.image_width

        if not os.path.exists(png_data_folder):
            self.report({'ERROR'}, "Invalid PNG data folder path")
            return {'CANCELLED'}

        for file_name in os.listdir(png_data_folder):
            if file_name.endswith('.png'):
                png_path = os.path.join(png_data_folder, file_name)
                exr_name = os.path.splitext(file_name)[0] + '.exr'
                exr_path = os.path.join(png_data_folder, exr_name)

                # Load PNG image
                image = bpy.data.images.load(png_path)

                # Set image dimensions
                image.scale(image_width, image_height)

                # Save as EXR
                image.filepath_raw = exr_path
                image.file_format = 'OPEN_EXR'
                image.save()

                # Unload image to free memory
                bpy.data.images.remove(image)

        self.report({'INFO'}, "PNG to EXR conversion complete.")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(GPTExrGeneratorProperties)
    bpy.utils.register_class(GPTExrGeneratorPanel)
    bpy.utils.register_class(GenerateRandomBrushLibOperator)
    bpy.utils.register_class(GenerateExrDataOperator)
    bpy.utils.register_class(ConvertPngToExrOperator)
    bpy.utils.register_class(CreateExrDataOperator)
    bpy.types.Scene.gpt_exr_generator = bpy.props.PointerProperty(type=GPTExrGeneratorProperties)

    
def unregister():
    bpy.utils.unregister_class(GPTExrGeneratorProperties)
    bpy.utils.unregister_class(GPTExrGeneratorPanel)
    bpy.utils.unregister_class(GenerateRandomBrushLibOperator)
    bpy.utils.unregister_class(GenerateExrDataOperator)
    bpy.utils.unregister_class(ConvertPngToExrOperator)
    bpy.utils.unregister_class(CreateExrDataOperator)
    del bpy.types.Scene.gpt_exr_generator

if __name__ == "__main__":
    register()

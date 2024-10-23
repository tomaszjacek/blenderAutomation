# Import necessary Blender Python modules and standard libraries

import bpy  # Main Blender Python API
import random  # For generating random positions
from bpy.props import IntProperty, StringProperty  # Property definitions for the addon
from bpy.types import Operator, Panel  # Base classes for operators and UI panels
from mathutils import Vector  # Blender's vector math library
import os  # For file path operations

# Addon metadata and registration information
bl_info = {
    "name": "Outlined Deeplearning Generator",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Outline Generator",
    "description": "Generate random object configurations and render images",
    "category": "3D View",
}

class OBJECT_OT_add_plane(Operator):
    """Operator to add a plane to the scene"""
    bl_idname = "object.add_plane"
    bl_label = "Add Plane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Add a plane at a specific location"""
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
        bpy.context.object.location[0] = 0
        bpy.context.object.location[1] = 0
        bpy.context.object.location[2] = 0
        bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

        return {'FINISHED'}

class OBJECT_OT_add_camera(Operator):
    """Operator to add a camera to the scene"""
    bl_idname = "object.add_camera"
    bl_label = "Add Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Add a camera at position (0, 0, 5)
        bpy.ops.object.camera_add(location=(0, 0, 5))
        camera = context.active_object
        camera.location.z = 5  # Ensure Z location is exactly 5
        bpy.context.object.rotation_euler[0] = 0
        bpy.context.object.rotation_euler[1] = 0
        bpy.context.object.rotation_euler[2] = 0


        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 6), scale=(1, 1, 1))

        return {'FINISHED'}

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b, 1

class OBJECT_OT_generate_data(Operator):
    """
    Operator to generate a collection of random 3D objects.

    This operator creates a new collection named "data" and populates it with
    a specified number of random objects (cone, sphere, torus, cube, cylinder).
    The number of objects is determined by the scene's 'number_of_objects' property.
    """
    bl_idname = "object.generate_data"
    bl_label = "Generate Data"
    bl_options = {'REGISTER', 'UNDO'}


    
    def execute(self, context):
        scene = context.scene

        if "data" in bpy.data.objects:
            bpy.data.collections.remove(bpy.data.collections["data"], do_unlink=True)

        # Create a new collection to hold the generated objects
        data_collection = bpy.data.collections.new("data")
        scene.collection.children.link(data_collection)

        # Define the types of objects to be generated
        object_types = [
            ('CONE', bpy.ops.mesh.primitive_cone_add),
            ('SPHERE', bpy.ops.mesh.primitive_uv_sphere_add),
            ('TORUS', bpy.ops.mesh.primitive_torus_add),
            ('CUBE', bpy.ops.mesh.primitive_cube_add),
            ('CYLINDER', bpy.ops.mesh.primitive_cylinder_add)
        ]
        object_count = 0

        try:
            
            while object_count < scene.number_of_objects:
                for obj_type, add_func in object_types:
                    if object_count >= scene.number_of_objects:
                        break
                    # Add the object to the scene
                    add_func()
                    obj = context.active_object
                    bpy.ops.transform.resize(value=(0.4783816, 0.4783816, 0.4783816), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
                    bpy.ops.object.material_slot_add()
                    mat = bpy.data.materials.new("My Material")
                    mat.diffuse_color = get_random_color()
                    #obj.data.materials.append(mat)
                    obj.material_slots[0].material = mat
                    # Ensure the object is linked to the scene collection
                    if obj.name not in scene.collection.objects:
                        scene.collection.objects.link(obj)

                    # Link the object to our data collection
                    if obj.name not in data_collection.objects:
                        data_collection.objects.link(obj)

                    # Unlink the object from the main scene collection
                    # This ensures it's only in our data collection
                    if obj.name in scene.collection.objects:
                        scene.collection.objects.unlink(obj)

                    object_count += 1

            self.report({'INFO'}, f"Successfully generated {object_count} objects.")
            return {'FINISHED'}
        except Exception as e:
            # If an error occurs during object generation, report it and cancel the operation
            self.report({'ERROR'}, f"Error generating objects: {str(e)}")
            return {'CANCELLED'}

class OBJECT_OT_generate_images(Operator):
    """
    Operator to generate a series of rendered images with randomized object positions.

    This operator uses the objects in the "data" collection and a camera to create
    a specified number of rendered images. For each render, the objects are
    repositioned randomly within a defined space.
    """
    bl_idname = "object.generate_images"
    bl_label = "Generate Images"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        data_collection = bpy.data.collections.get("data")
        camera = bpy.data.objects.get("Camera")

        train_path = os.path.join(scene.photo_path, "train")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        trainx_path = os.path.join(train_path, "x")
        if not os.path.exists(trainx_path):
            os.makedirs(trainx_path)
        trainy_path = os.path.join(train_path, "y")
        if not os.path.exists(trainy_path):
            os.makedirs(trainy_path)

        # Check if required objects exist
        if not data_collection or not camera:
            self.report({'ERROR'}, "Data collection or camera not found")
            return {'CANCELLED'}

        # Generate images for the specified number of steps
        for step in range(scene.number_of_steps):
            # Randomize object positions for each render
            for obj in data_collection.objects:
                obj.location = Vector((
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(1, 5)
                ))

            # Set up the render
            scene.camera = camera
            filepath = ""
            #bpy.context.space_data.context = 'TOOL'
            #bpy.context.space_data.context = 'RENDER'
            bpy.context.scene.render.use_freestyle = True
            #bpy.context.space_data.context = 'VIEW_LAYER'
            bpy.context.scene.view_layers["ViewLayer"].use_solid = False

            filepath = os.path.join(trainy_path, f"render_{step:03d}.png")
            scene.render.filepath = filepath
            # Perform the render and save the image
            bpy.ops.render.render(write_still=True)


            bpy.context.scene.render.use_freestyle = False
            #bpy.context.space_data.context = 'VIEW_LAYER'
            bpy.context.scene.view_layers["ViewLayer"].use_solid = True
            filepath = os.path.join(trainx_path, f"render_{step:03d}.png")
            scene.render.filepath = filepath
            # Perform the render and save the image
            bpy.ops.render.render(write_still=True)

        return {'FINISHED'}

class VIEW3D_PT_devin_generator(Panel):
    """
    Panel class for the Devin Generator addon.
    This creates the user interface in the 3D Viewport's sidebar.
    """
    bl_space_type = 'VIEW_3D'  # Appear in the 3D Viewport
    bl_region_type = 'UI'  # Appear in the sidebar
    bl_category = "Devin Generator"  # Name of the tab in the sidebar
    bl_label = "Devin Generator"  # Label at the top of the panel

    def draw(self, context):
        """
        Define the layout of the panel.
        This method is called every time the panel is drawn.
        """
        layout = self.layout
        scene = context.scene

        # Add buttons and properties to the layout
        layout.operator("object.add_plane", text="Add Plane")
        layout.prop(scene, "number_of_objects")
        layout.operator("object.generate_data", text="Generate Data")
        layout.operator("object.add_camera", text="Add Camera")
        layout.prop(scene, "photo_path")
        layout.prop(scene, "number_of_steps")
        layout.operator("object.generate_images", text="Generate")

# Tuple of all classes in the addon for registration
classes = (
    OBJECT_OT_add_plane,
    OBJECT_OT_add_camera,
    OBJECT_OT_generate_data,
    OBJECT_OT_generate_images,
    VIEW3D_PT_devin_generator,
)

def register():
    """
    Register all classes and properties for the addon.
    This function is called when the addon is enabled in Blender.
    """
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register custom properties
    bpy.types.Scene.number_of_objects = IntProperty(
        name="Number of Objects",
        default=10,
        min=1,
        description="Number of objects to generate"
    )
    bpy.types.Scene.photo_path = StringProperty(
        name="Photo Path",
        default="//",
        subtype='DIR_PATH',
        description="Path to save rendered images"
    )
    bpy.types.Scene.number_of_steps = IntProperty(
        name="Number of Steps",
        default=10,
        min=1,
        description="Number of steps for image generation"
    )

def unregister():
    """
    Unregister all classes and remove properties.
    This function is called when the addon is disabled in Blender.
    """
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove custom properties
    del bpy.types.Scene.number_of_objects
    del bpy.types.Scene.photo_path
    del bpy.types.Scene.number_of_steps

# This allows you to run the script directly from Blender's Text editor
if __name__ == "__main__":
    register()

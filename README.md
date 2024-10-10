# Devin Outlined Deeplearning Generator

## Installation

1. Download the `devinOutlinedDeeplearningGenerator.py` file.
2. Open Blender and go to Edit > Preferences > Add-ons.
3. Click "Install" and navigate to the downloaded file.
4. Select the file and click "Install Add-on".
5. Enable the add-on by checking the box next to "3D View: Devin Outlined Deeplearning Generator".

## Usage

1. After installation, you'll find the "Devin Generator" panel in the 3D Viewport's sidebar (press N to toggle the sidebar if it's not visible).

2. The panel contains the following options:

   - Add Plane: Adds a plane to the scene at location (7, 7, 0).
   - Number of Objects: Set the number of objects to generate (default: 10).
   - Generate Data: Creates a collection named "data" and populates it with random objects (cone, sphere, torus, cube, cylinder) based on the "Number of Objects" input.
   - Add Camera: Adds a camera to the scene at location (0, 0, 5).
   - Photo Path: Set the directory where rendered images will be saved.
   - Number of Steps: Set the number of image renders to generate (default: 10).
   - Generate: Generates the specified number of rendered images.

3. Workflow:
   a. Click "Add Plane" to add a base plane to your scene.
   b. Set the desired number of objects.
   c. Click "Generate Data" to create the object collection.
   d. Click "Add Camera" to add a camera to the scene.
   e. Set the photo path where you want to save the rendered images.
   f. Set the number of steps (renders) you want to generate.
   g. Click "Generate" to create the renders.

4. The addon will create a series of renders, each with randomly positioned objects, saved as PNG files in the specified photo path.

## Notes

- Ensure you have write permissions for the directory specified in the "Photo Path".
- The generated images will be named "render_000.png", "render_001.png", etc.
- Each render will have the objects in the "data" collection randomly positioned.

For any issues or feature requests, please contact the addon author.

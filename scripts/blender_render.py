"""Blender script to render images of 3D models.
This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.
Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2
Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""
import argparse
import glob
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import numpy as np
import scipy.stats as stats
import bpy
from mathutils import Vector, Matrix
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True,
                    help="Path to the object file")
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument("--anno_dir", type=str, default="./views")
parser.add_argument("--depth_dir", type=str, default="./views")
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--engine", type=str, default="BLENDER_EEVEE",
                    choices=["CYCLES", "BLENDER_EEVEE"])

parser.add_argument("--distance", type=float, default=1.5)
parser.add_argument("--azimuth", type=float, default=0.0, help='degrees')
parser.add_argument("--elevation", type=float, default=0.0, help='degrees')
parser.add_argument("--pose_sampling", type=int, default=1)
parser.add_argument("--strength", type=int, default=1)

parser.add_argument("--distance_min", type=float, default=0.5, help="min = args.distance - args.distance_min")
parser.add_argument("--distance_max", type=float, default=1.5, help="max = args.distance + args.distance_max")

parser.add_argument("--bg_hdr_dir", type=str, default=None)

parser.add_argument("--depth_format", type=str, default="OPEN_EXR",
                    choices=["PNG", "OPEN_EXR"])

parser.add_argument("--disable_random", action="store_true")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

if args.distance_min + args.distance_max < 1e-5:
    disable_distance_sample = True
else:
    disable_distance_sample = False
    args.distance_min = args.distance - args.distance_min
    args.distance_max = args.distance + args.distance_max
    args.distance_mean = args.distance
    args.distance_var = 0.5

if args.pose_sampling == 1:
    # viewing from all azimuth angles, and all elevation angles, e.g., airliner.
    args.azimuth_min, args.azimuth_max = 0, 360
    args.elevation_mean, args.elevation_var = 90, 30
    args.elevation_min, args.elevation_max = 0, 180
elif args.pose_sampling == 2:
    # viewing from front azimuth angles, and all elevation angles, e.g., microwave, washing machine.
    args.azimuth_min, args.azimuth_max = -195, 15
    args.elevation_mean, args.elevation_var = 90, 30
    args.elevation_min, args.elevation_max = 0, 180
elif args.pose_sampling == 3:
    # viewing from all azimuth angles, and only elevation angles when viewing from up to down, e.g., keyboard, bath tub, car, beach wagon.
    args.azimuth_min, args.azimuth_max = 0, 360
    args.elevation_mean, args.elevation_var = 60, 20
    args.elevation_min, args.elevation_max = 0, 195
elif args.pose_sampling == 4:
    # viewing from front azimuth angles, and only elevation angles when viewing from up to down, e.g., printer, stove.
    args.azimuth_min, args.azimuth_max = -195, 15
    args.elevation_mean, args.elevation_var = 60, 20
    args.elevation_min, args.elevation_max = 0, 195
elif args.pose_sampling == 5:
    args.azimuth_min, args.azimuth_max = 0, 360
    args.elevation_mean, args.elevation_var = None, None
    args.elevation_min, args.elevation_max = 0, 180
else:
    raise ValueError(f'Unknown pose sampling strategy {args.pose_sampling}')

context = bpy.context
scene = context.scene
render = scene.render
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100
scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    try:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    except:
        pass
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.object.select_all(action='DESELECT')
        this_obj = bpy.ops.import_scene.obj(filepath=object_path, use_edges=False, use_smooth_groups=False, split_mode='OFF')
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                this_obj.select_set(True)
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode='OBJECT')
    elif 'toys4k' in object_path and object_path.endswith('.blend'):
        with bpy.data.libraries.load(object_path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
        for obj in data_to.objects:
            if obj is not None:
                scene.collection.objects.link(obj)

        obj = [obj for obj in bpy.data.objects if (obj.name != "Camera")][0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(-90), 0, 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    return cam


def get_random_dist(args):
    if disable_distance_sample:
        return args.distance
    else:
        l = (args.distance_min - args.distance_mean) / args.distance_var
        r = (args.distance_max - args.distance_mean) / args.distance_var
        return stats.truncnorm.rvs(l, r, loc=args.distance_mean, scale=args.distance_var)


def get_random_pose(args):
    # theta: 0 (right), 90 (front)
    theta = np.random.rand() * (args.azimuth_max - args.azimuth_min) + args.azimuth_min
    if args.elevation_mean is not None and args.elevation_var is not None:
        l = (args.elevation_min - args.elevation_mean) / args.elevation_var
        r = (args.elevation_max - args.elevation_mean) / args.elevation_var
        phi = stats.truncnorm.rvs(l, r, loc=args.elevation_mean, scale=args.elevation_var)
    else:
        phi = np.random.rand() * (args.elevation_max - args.elevation_min) + args.elevation_min
    # l = (args.roll_min - args.roll_mean) / args.roll_var
    # r = (args.roll_max - args.roll_mean) / args.roll_var
    # camera_rotation = stats.truncnorm.rvs(l, r, loc=args.roll_mean, scale=args.roll_var)
    camera_rotation = 0.0
    return math.radians(theta), math.radians(phi), math.radians(camera_rotation)


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.anno_dir, exist_ok=True)
    os.makedirs(args.depth_dir, exist_ok=True)
    reset_scene()

    if args.bg_hdr_dir is not None:
        bg_hdrs = glob.glob(os.path.join(args.bg_hdr_dir, '**/*.hdr'), recursive=True)
        np.random.shuffle(bg_hdrs)
        bpy.context.scene.render.film_transparent = False
        world = bpy.context.scene.world
        node_tree = world.node_tree
        output_node = world.node_tree.get_output_node('CYCLES')
        [node_tree.links.remove(x) for x in output_node.inputs[0].links]
        background_node = node_tree.nodes.new(type="ShaderNodeBackground")
        node_tree.links.new(background_node.outputs[0], output_node.inputs[0])
        img = bpy.data.images.load(bg_hdrs[0])
        env_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        env_texture_node.image = img
        node_tree.links.new(env_texture_node.outputs[0], background_node.inputs[0])

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    _scale, _offset = normalize_scene()
    add_lighting()
    cam = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)

    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    render_layers = nodes.new('CompositorNodeRLayers')

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = args.depth_format
    if args.depth_format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_depth = "8"
        depth_file_output.format.color_mode = "BW"
        map = nodes.new(type="CompositorNodeMapValue")
        map.offset = [-0.7]
        map.size = [1.0]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

    if args.disable_random:
        num_images = 1
    else:
        num_images = args.num_images

    for i in range(num_images):
        if args.bg_hdr_dir is not None:
            img = bpy.data.images.load(bg_hdrs[i])
            env_texture_node.image = img

        image_file = os.path.join(args.output_dir, f"{i:03d}.png")
        annotation_file = os.path.join(args.anno_dir, f'{i:03d}.npy')
        depth_file = os.path.join(args.depth_dir, f"{i:03d}.exr")
        if os.path.isfile(image_file) and os.path.isfile(annotation_file) and os.path.isfile(depth_file):
            continue

        if os.path.isfile(image_file):
            os.remove(image_file)
        if os.path.isfile(annotation_file):
            os.remove(annotation_file)
        if os.path.isfile(depth_file):
            os.remove(depth_file)

        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_constraint.target = empty
        if not args.disable_random:
            theta, phi, camera_rotation = get_random_pose(args)
            camera_dist = get_random_dist(args)
        else:
            theta, phi, camera_rotation = -np.pi/2, np.pi/3, 0.0
            camera_dist = 1.5

        theta = theta + args.azimuth / 180.0 * np.pi
        phi = phi - args.elevation / 180.0 * np.pi

        phi = phi % (2 * np.pi)
        if phi > np.pi:
            theta = theta + np.pi
            phi = 2 * np.pi - phi
        theta = theta % (2 * np.pi)
        phi = np.clip(phi, 0, np.pi)
        camera_rotation = np.clip(camera_rotation, -np.pi, np.pi)
        if phi == 0.0:
            phi = 1e-5
        # set the camera position
        point = (
            camera_dist * math.sin(phi) * math.cos(theta),
            camera_dist * math.sin(phi) * math.sin(theta),
            camera_dist * math.cos(phi),
        )
        cam.location = point
        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)
        bpy.context.view_layer.objects.active = cam
        bpy.ops.object.visual_transform_apply()
        bpy.ops.object.constraints_clear()
        bpy.ops.object.select_all(action='DESELECT')
        location = cam.location.copy()
        right, up, back = cam.matrix_world.to_3x3().transposed()
        direction = np.cross(up,right)
        rotation_vertical = Matrix.Rotation(camera_rotation, 3, Vector(direction))
        matrix = rotation_vertical @ cam.matrix_world.to_3x3()
        cam.matrix_world = matrix.to_4x4()
        cam.location = location
        # render the image
        render_path = os.path.join(args.output_dir, f"{i:03d}.png")
        depth_path = os.path.join(args.depth_dir, f"{i:03d}")
        scene.render.filepath = render_path
        depth_file_output.file_slots[0].path = depth_path

        bpy.ops.render.render(write_still=True)
        _filename = os.path.basename(args.object_path)
        _src = 'objaverse' if _filename.endswith('.glb') else 'shapenet'
        if _src == 'objaverse':
            _shape_id = '.'.join(_filename.split('.')[:-1])
        else:
            _shape_id = args.object_path.split('/')[-3]

        mat = cam.matrix_world
        matrix_world = np.array([list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])])

        mat = cam.matrix_world.inverted()
        modelview_matrix = np.array([list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])])

        mat = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=render.resolution_x, y=render.resolution_y, scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y)
        projection_matrix = np.array([list(mat[0]), list(mat[1]), list(mat[2]), list(mat[3])])

        pose = {
            'theta': theta,
            'phi': phi,
            'dist': camera_dist,
            'camera_rotation': camera_rotation,
            'principal': np.array([256, 256]),
            'source': _src,
            'shape_id': _shape_id,
            'rendering_scale': _scale,
            'rendering_offset': np.array(_offset),
            'azimuth': args.azimuth,
            'elevation': args.elevation,
            'pose_sampling': args.pose_sampling,
            'strength': args.strength,
            'matrix_world': matrix_world,
            'modelview_matrix': modelview_matrix,
            'projection_matrix': projection_matrix,
            'focal_length': cam.data.lens,
            'sensor_width': cam.data.sensor_width,
            'sensor_height': cam.data.sensor_height
        }
        np.save(os.path.join(args.anno_dir, f'{i:03d}'), pose)
        shutil.move(depth_path+'0001.exr', depth_path+'.exr')
    # bpy.ops.wm.save_as_mainfile(filepath='/data/home/wufeim/dst_cvpr/dst_render_bg3d/main.blend')


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
